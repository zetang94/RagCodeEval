import math
import re
import warnings
from collections import defaultdict

from fuzzywuzzy import fuzz
from torch.utils.data import Dataset

from tqdm import tqdm
import json
import torch

from codeprep.tokens.containers import SplitContainer
import codeprep.api.text as cp


class TokenDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['task_id'])

    def __getitem__(self, idx):
        return {
            'task_id': self.data['task_id'][idx],
            'input_ids': torch.LongTensor(self.data['input_ids'][idx]),
            'input_len': self.data['input_len'][idx],
        }


class NextLineCompletionTask:
    def __init__(self, args, tokenizer, n_tasks=None, stop_words=['\n']):
        self.args = args
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.n_copies = None

        samples = []

        try:
            with open(args.test_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f.readlines(), f'Load dataset from {args.test_file}.'):
                    sample = json.loads(line)
                    if 'crossfile_context' not in sample:
                        print("context does not have cross context file.")
                        continue
                    samples.append(sample)
        except Exception as e:
            print(e)

        if n_tasks is not None:
            samples = samples[:n_tasks]

        self.dataset = samples

    def get_dataset(self):
        return self.dataset

    def tokenize(self, n_copies=1, num_devices=1):
        self.n_copies = n_copies

        max_prompt_length = self.args.max_sequence_len - self.args.max_gen_len

        crossfile_context = []

        for example in self.dataset:
            cfc_text, actual_shot_num = self.k_shot_examples(example)
            example['actual_shot_num'] = actual_shot_num

            crossfile_context.append(cfc_text)

        if self.args.k_shots > 0:
            self.tokenizer.truncation_side = "right"
            crossfile_features = self.tokenizer(
                crossfile_context,
                truncation=True,
                max_length=self.args.max_cfg_len
            )

        features = {"input_ids": [], "attention_mask": []}
        self.tokenizer.truncation_side = "left"
        task_ids = []

        for idx, example in enumerate(self.dataset):
            prompt = example['prompt']
            allowed_prompt_length = max_prompt_length
            if self.args.k_shots > 0:
                allowed_prompt_length = allowed_prompt_length - len(crossfile_features["input_ids"][idx])

            prompt_feats = self.tokenizer(
                [prompt],
                truncation=True,
                max_length=allowed_prompt_length
            )

            for _ in range(n_copies):
                for k, v in prompt_feats.items():
                    if k not in features.keys():
                        continue

                    if self.args.k_shots > 0:
                        features[k].append(crossfile_features[k][idx] + prompt_feats[k][0])
                    else:
                        features[k].append(prompt_feats[k][0])

                # Task Id = example['idx']
                task_ids.append(example['idx'])

        remainder = len(features["input_ids"]) % num_devices

        if remainder > 0:
            for _ in range(num_devices - remainder):
                features["input_ids"].append(features["input_ids"][-1])
                features['attention_mask'].append(features['attention_mask'][-1])
                task_ids.append(task_ids[-1])

        # pad to max_seq_length
        self.tokenizer.padding_side = "right"
        features = self.tokenizer.pad(features, padding="max_length",
                                      max_length=max_prompt_length)

        features['input_len'] = [sum(mask) for mask in features['attention_mask']]
        features['task_id'] = task_ids

        return TokenDataset(features)

    def k_shot_examples(self, example):
        retrieval_chunks = example["crossfile_context"]["list"]

        ls_sym = "#" if self.args.language == 'Python' else "//"
        cfc_text = ""
        if retrieval_chunks and self.args.k_shots > 0:
            init_cfc_text = f"{ls_sym} Here are some relevant code fragments from other files of the repo:\n\n"
            cfc_length = len(self.tokenizer.tokenize(init_cfc_text))
            num_chunk_inc = 0

            for cfc_idx, cfc_chunk in enumerate(retrieval_chunks[:self.args.k_shots]):
                add_text = f"{ls_sym} the below code fragment is found in {cfc_chunk['filename']}" + "\n"
                cfc_lines = cfc_chunk["retrieved_chunk"].split('\n')
                add_text += "\n".join([f"{ls_sym} {cl}" for cl in cfc_lines if cl]) + "\n\n"
                # check if adding chunk exceeds max length budget for CFC
                add_text_len = len(self.tokenizer.tokenize(add_text))
                if cfc_length + add_text_len <= self.args.max_cfg_len:
                    cfc_text += add_text
                    cfc_length += add_text_len
                    num_chunk_inc += 1
                else:
                    break
            if num_chunk_inc > 0:
                cfc_text = init_cfc_text + cfc_text

        return cfc_text, num_chunk_inc

    def get_prompt(self, doc):
        return doc['prompt']

    def get_reference(self, doc):
        return doc['ground_truth']

    def get_actual_shot_num(self, doc):
        if 'actual_shot_num' in doc:
            return doc['actual_shot_num']
        else:
            return self.args.k_shots

    @staticmethod
    def _stop_at_stop_token(decoded_string, stop_tokens):
        """
        Produces the prefix of decoded_string that ends at the first occurrence of
        a stop_token.
        WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
        itself.
        """
        min_stop_index = len(decoded_string)
        for stop_token in stop_tokens:
            stop_index = decoded_string.find(stop_token)
            if -1 < stop_index < min_stop_index:
                min_stop_index = stop_index
        return decoded_string[:min_stop_index]

    def post_process(self, generation):
        # remove comments
        generation = re.sub(r'#.*', '', generation)
        generation = re.sub(r'//.*', '', generation)
        generation = generation.strip('\n ')

        return self._stop_at_stop_token(generation, self.stop_words)

    def process_results(self, generations, references, dataset):
        edit_sim = 0.0
        em = 0.0
        em_id = 0.0
        f1_id = 0.0

        results = {'statistic': []}

        for (gens, gt, example) in zip(generations, references, dataset):
            if gt.endswith('\n'):
                gt = gt[:-1]

            example_statistic = {
                'id': example['idx'],
                'lifecycle': example['file_meta']['lifecycle'],
                'hasRef': len(example['history_callees']) > 0,
                'hasDef': len(example['definitions']) > 0,
                'em': [],
                'es': [],
                'emId': [],
                'f1Id': []
            }

            if 'actual_shot_num' in example:
                hit = False
                shot_num = example["actual_shot_num"]
                for cft in example['crossfile_context']['list'][:shot_num]:
                    if cft['hit']:
                        hit = True
                        break
                example_statistic['hit'] = hit

            gt_ids = get_identifier(gt, self.args.language)

            # Only Compute max value.
            cur_max_es = 0
            cur_max_em = 0
            cur_max_em_id = 0
            cur_max_f1_id = 0

            for pred in gens[:self.args.n_samples]:
                cur_es = fuzz.ratio(pred, gt)
                cur_em = 1 if pred == gt else 0

                pred_ids = get_identifier(pred, self.args.language)
                cur_em_id = 0
                if gt_ids == pred_ids:
                    cur_em_id = 1

                id_tp, id_fp, id_fn = compute_id_match(pred_ids, gt_ids)
                cur_f1 = 0
                if (2 * id_tp + id_fp + id_fn) != 0:
                    cur_f1 = 2 * id_tp / (2 * id_tp + id_fp + id_fn)

                if cur_es > cur_max_es:
                    cur_max_es = cur_es
                if cur_em > cur_max_em:
                    cur_max_em = cur_em
                if cur_em_id > cur_max_em_id:
                    cur_max_em_id = cur_em_id
                if cur_f1 > cur_max_f1_id:
                    cur_max_f1_id = cur_f1

                example_statistic['em'].append(cur_em)
                example_statistic['es'].append(cur_es)
                example_statistic['emId'].append(cur_em_id)
                example_statistic['f1Id'].append(cur_f1)

            edit_sim += cur_max_es
            em += cur_max_em
            em_id += cur_max_em_id
            f1_id += cur_max_f1_id

            results['statistic'].append(example_statistic)

        results['mean'] = {'em': em * 100 / len(references),
                           'es': edit_sim / len(references),
                           'emId': em_id * 100 / len(references),
                           'f1Id': f1_id * 100 / len(references),
                           }

        return results


def get_identifier(code, lang):
    if lang == 'Java':
        lang = "java"
    elif lang == 'Python':
        lang = 'python'

    no_spaces = True if lang == 'java' else False
    identifiers = []
    tokens, metadata = cp.nosplit(code,
                                  extension=lang,
                                  no_spaces=no_spaces,
                                  no_unicode=True,
                                  no_com=True,
                                  full_strings=True,
                                  max_str_length=15,
                                  return_metadata=True)

    token_types = list(map(lambda x: x.__name__, metadata.token_types))

    for i, (token, token_type) in enumerate(zip(tokens, token_types)):
        if token_type in [SplitContainer.__name__]:
            identifiers.append(token)

    return identifiers


def compute_id_match(pred_ids, target_ids):
    pred_ids = list(set(pred_ids))
    target_ids = list(set(target_ids))
    tp = 0
    fp = 0
    fn = 0
    for pid in pred_ids:
        if pid in target_ids:
            tp += 1
        else:
            fp += 1
    for tid in target_ids:
        if tid not in pred_ids:
            fn += 1
    return tp, fp, fn


def complete_code(
    task,
    accelerator,
    model,
    tokenizer,
    dataloader,
    n_tasks,
    n_copies,
    batch_size=20,
    prefix="",
    postprocess=True,
    **gen_kwargs,
):
    """Generate multiple codes for each task in the dataset using multiple GPUs with accelerate.
    dataloader sends all the prompts from the evalution dataset to the model as the following:
    [p_0_0, p_0_1, ..., p_0_nc-1, p_1_0, ..., p_nt-1_nc-1] where nc is the number of copies of the prompt,
    and nt is the number of tasks. nc is such that num_samples(for each task)= nc * batch_size
    """

    gen_token_dict = defaultdict(list)  # dict of list of generated tokens
    for step, batch in tqdm(
        enumerate(dataloader), total=math.ceil(n_tasks * n_copies / accelerator.state.num_processes),
    ):
        with torch.no_grad():
            if task.stop_words:
                # Set the start_length after which to check for stopping to be the longest input ignoring padding
                gen_kwargs["stopping_criteria"][0].start_length = (
                    batch["input_len"].max().item()
                )

            generated_tokens = model.generate(
                input_ids=batch["input_ids"][:, : batch["input_len"]],
                num_return_sequences=batch_size,
                **gen_kwargs,
            )
            # each task is generated batch_size times
            generated_tasks = batch["task_id"].repeat(batch_size)

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens, generated_tasks = accelerator.gather(
                (generated_tokens, generated_tasks)
            )

            # Without prompt outputs
            generated_tokens = generated_tokens[:, batch["input_len"][0]:]

            generated_tokens = generated_tokens.cpu().numpy()
            generated_tasks = generated_tasks.cpu().numpy()

            for sample, generated_seqs in zip(generated_tasks, generated_tokens):
                gen_token_dict[sample].append(generated_seqs)

    task_ids = sorted(gen_token_dict.keys())
    code_gens = [[] for _ in range(n_tasks)]
    for sample, generated_tokens in gen_token_dict.items():
        for s in generated_tokens:
            if tokenizer.eos_token in task.stop_words:
                gen_code = tokenizer.decode(
                    s, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
            else:
                if tokenizer.name_or_path in ["facebook/incoder-1B", "facebook/incoder-6B"]:
                    # The same setting as in https://huggingface.co/facebook/incoder-6B
                    gen_code = tokenizer.decode(
                        s, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                else:
                    gen_code = tokenizer.decode(
                        s, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )

            gen_code = gen_code[len(prefix):]
            if postprocess:
                code_gens[task_ids.index(sample)].append(
                    task.post_process(gen_code)
                )
            else:
                warnings.warn(
                    "model output is not postprocessed, this might lower evaluation scores"
                )
                code_gens[task_ids.index(sample)].append(gen_code)

    return code_gens
