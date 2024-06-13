import json
import re
from math import ceil
import torch
import numpy as np
from accelerate.utils import set_seed
from torch.utils.data import SequentialSampler
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList

from code_completion import complete_code


def custom_data_collator(features):
    first = features[0]
    batch = {}
    for k, v in first.items():
        if v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
        if v is not None and isinstance(v, str):
            batch[k] = [f[k] for f in features]

    return batch


class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""

        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length:]
        )

        done = []
        for decoded_generation in decoded_generations:
            # delete comments.
            decoded_generation = re.sub(r'#.*', '', decoded_generation)
            decoded_generation = re.sub(r'//.*', '', decoded_generation)
            decoded_generation = decoded_generation.lstrip()

            done.append(
                any(
                    [
                        stop_string in decoded_generation and decoded_generation
                        for stop_string in self.eof_strings
                    ]
                )
            )
        return all(done)


def parallel_generations(task, tokenized_dataset, accelerator, model, tokenizer, n_tasks, n_copies, args):
    if args.load_generations_path:
        # load generated code
        with open(args.load_generations_path) as fp:
            generations = json.load(fp)
            if accelerator.is_main_process:
                print(
                    f"generations loaded, {n_tasks} selected from {len(generations)} with {len(generations[0])} candidates"
                )
        return generations[:n_tasks]

    set_seed(args.seed, device_specific=True)

    # Setup generation settings
    gen_kwargs = {
        "do_sample": args.do_sample,  # 不一定选择最高的概率
        "temperature": args.temperature,  # 作用于softmax，tmp越大，越平滑，越小，越shape
        "top_p": args.top_p,  # 在总和满足p中选择
        "top_k": args.top_k,  # 在总数满足k中选择
        "max_length": args.max_sequence_len,  # 最大的生成长度，包含prompt
    }
    if task.stop_words:
        if tokenizer.eos_token:
            task.stop_words.append(tokenizer.eos_token)
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [EndOfFunctionCriteria(0, task.stop_words, tokenizer)]
        )

    if accelerator.is_main_process:
        print(f"number of problems for this task is {n_tasks}")

    # do not confuse args.batch_size, which is actually the num_return_sequences
    ds_loader = DataLoader(
        tokenized_dataset,
        batch_size=1
    )

    model = model.to(accelerator.device)
    ds_loader = accelerator.prepare(ds_loader)

    generations = complete_code(
        task,
        accelerator,
        model,
        tokenizer,
        ds_loader,
        n_tasks=n_tasks,
        n_copies=n_copies,
        batch_size=args.batch_size,
        prefix=args.prefix,
        postprocess=args.postprocess,
        **gen_kwargs,
    )
    return generations
