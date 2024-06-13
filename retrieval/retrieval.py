"""
The code is modified from
https://github.com/amazon-science/cceval/blob/main/prompt_builder/augment_with_cfc.py
"""
import argparse
import glob
import json
import math
import os
import multiprocessing as mp
import random
import time
from functools import partial

import jsonlines
import numpy as np
import torch.cuda
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from rerank_utils import SemanticReranking, lexical_ranking
from ranking_metrics import evaluate_retrieval_metric
from ftp_utils import ftpconnect, uploadfile
from utils import str2bool, tokenize_nltk, find_files_within_distance_k


CHUNK_SIZE = 10
SLIDING_WINDOW_SIZE = 10  # non-overlapping chunks if SLIDING_WINDOW_SIZE=CHUNK_SIZE
QUERY_LENGTH = 10  # last N lines from prompt will be queried
CODE_BLOCK_STATUS = {"NO_DEF_NO_REF": 0, "HAS_DEF": 1, "HAS_REF": 2, "BOTH": 3}

file_ext = {"Python": "py", "Java": "java", "TypeScript": "ts"}


class CodeRetrieval:
    def __init__(self, input_args, repository_root=None, test_samples=None, use_codalab=True):
        self.args = input_args
        self.repository_root = repository_root
        self.test_samples = test_samples
        self.test_size = len(test_samples)
        self.use_codalab = use_codalab

        self.both_scores = {'idx': []}
        self.ref_scores = {'idx': []}
        self.def_scores = {'idx': []}
        self.union_scores = {'idx': [], 'ref': {}, 'def': {}}

    def save_samples(self, output_examples, output_path, output_filename):
        # save test samples into smaller chunks if using codalab.
        file_path = os.path.join(output_path, output_filename)
        with open(file_path, "w") as fw:
            for ex in output_examples:
                fw.write(json.dumps(ex))
                fw.write("\n")

        if self.use_codalab:
            # Use ftp server to save output files.
            print("Beginning File Transfer.")
            ftp = ftpconnect("0.0.0.0", "user", "passwd")
            uploadfile(ftp, f"/retrieval/{output_filename}",
                       file_path)
            ftp.quit()
            print("END Transfer.")
            os.remove(file_path)

    def retrieve(self, output_path, output_filename):
        output_examples = self.build_retrieve_data()
        self.save_samples(output_examples, output_path, output_filename)

        union_len = len(self.union_scores['idx'])
        ref_len = len(self.ref_scores['idx'])
        def_len = len(self.def_scores['idx'])

        results = {
            'all': {
                'both': self.both_scores,
                'ref': self.ref_scores,
                'def': self.def_scores,
                'union': self.union_scores,
            },
            'mean': {
                'both': {k: np.mean(v) for k, v in self.both_scores.items()},
                'ref': {k: np.mean(v) for k, v in self.ref_scores.items()},
                'def': {k: np.mean(v) for k, v in self.def_scores.items()},
                'unionRef': {k: np.mean(v) for k, v in self.union_scores['ref'].items()},
                'unionDef': {k: np.mean(v) for k, v in self.union_scores['def'].items()}
            },
            'statistic': {
                'ref': ref_len,
                'def': def_len,
                'union': union_len
            },
            'args': {
                "lang": self.args.language,
                "ranking": self.args.ranking_fn,
                "concat": self.args.concat_strategy,

            }
        }

        if self.use_codalab:
            # Using codalab, score_file_name should be same,
            # then the schema can read them and display.
            score_file_name = "scores.json"
        else:
            score_file_name = output_filename.replace(".jsonl", "_score.jsonl")

        with open(os.path.join(output_path, score_file_name), "w") as fw:
            json.dump(results, fw)

    def build_retrieve_data(self):
        repositories = dict()  # Save repository code files.

        #output_examples = [None] * self.test_size

        output_example_dict = {}


        error_freq = {
            "project_not_found": 0,
            "no_cross_file_context": 0,
        }

        skip_samples_for_reason_ability = 0

        for sample in self.test_samples:
            repo_name = sample["file_meta"]["repo_name"]
            lifecycle = sample["file_meta"]["lifecycle"]
            repo_path = self.get_repo_path(repo_name, lifecycle)
            #root_path = os.path.join(self.repository_root, repo_path)

            sample['file_meta']['repo_path'] = repo_path
            file_path = sample['file_meta']['file_path']

            file_path = os.path.join(repo_path, file_path)
            sample['file_meta']['file_path'] = file_path

            if repo_path not in repositories:
                repositories[repo_path] = self.read_repo_files(repo_path)

        semantic_ranker = None
        if self.args.ranking_fn == "cosine_sim":
            semantic_ranker = SemanticReranking(
                self.args.ranker,
                max_sequence_length=256
            )

        pool = mp.Pool(self.args.num_processes)
        worker = partial(get_cfc,
                         args=self.args,
                         semantic_ranker=semantic_ranker,
                         repositories=repositories)

        with tqdm(total=len(self.test_samples)) as p_bar:
            for (d, stat) in pool.imap_unordered(worker, self.test_samples):

                if isinstance(stat, str):
                    if stat in ['project_not_found', 'no_cross_file_context']:
                        error_freq[stat] += 1
                        continue
                    if stat in ['not_both_ref_and_def']:
                        skip_samples_for_reason_ability += 1
                        continue
                else:
                    if 'scores' in stat:
                        ref_scores = stat['scores']['ref']
                        def_scores = stat['scores']['def']
                        both_scores = stat['scores']['both']

                        if ref_scores is not None:
                            self.ref_scores['idx'].append(d['idx'])
                            for m, n in ref_scores.items():
                                if m not in self.ref_scores:
                                    self.ref_scores[m] = []
                                self.ref_scores[m].append(n)

                        if def_scores is not None:
                            self.def_scores['idx'].append(d['idx'])
                            for m, n in def_scores.items():
                                if m not in self.def_scores:
                                    self.def_scores[m] = []
                                self.def_scores[m].append(n)

                        if both_scores is not None:
                            self.both_scores['idx'].append(d['idx'])
                            for m, n in both_scores.items():
                                if m not in self.both_scores:
                                    self.both_scores[m] = []
                                self.both_scores[m].append(n)

                        if ref_scores is not None and def_scores is not None:
                            self.union_scores['idx'].append(d['idx'])
                            for m, n in ref_scores.items():
                                if m not in self.union_scores['ref']:
                                    self.union_scores['ref'][m] = []
                                self.union_scores['ref'][m].append(n)

                            for m, n in def_scores.items():
                                if m not in self.union_scores['def']:
                                    self.union_scores['def'][m] = []
                                self.union_scores['def'][m].append(n)

                    if len(d["crossfile_context"]) == 0:
                        print("ERROR!, CrossFileContext len is zero!")
                    #output_examples.append(d)
                output_example_dict[d['idx']] = d
                p_bar.update()

        if skip_samples_for_reason_ability != 0:
            print(f'Skip {skip_samples_for_reason_ability} samples, '
                  f'{len(output_example_dict)} samples kept for reason ability test.')

        sorted_examples = [output_example_dict[key] for key in sorted(output_example_dict)]
        return sorted_examples

    @staticmethod
    def get_repo_path(repo_name, lifecycle):
        repo_path = os.path.join(
                                 repo_name.replace("/", "@"),
                                 lifecycle
                                 )

        return repo_path

    def read_repo_files(self, repo_path):
        # root_dir needs a trailing slash (i.e. /root/dir/)
        project_context = {}
        root_dir = os.path.join(self.repository_root, repo_path)

        if not os.path.isdir(root_dir):
            print(f"Repository not found: {root_dir}")
            return project_context

        if self.args.language == "TypeScript":
            src_files = []
            src_files += glob.glob(os.path.join(root_dir, f'**/*.ts'), recursive=True)
            src_files += glob.glob(os.path.join(root_dir, f'**/*.tsx'), recursive=True)
        else:
            src_files = glob.glob(os.path.join(root_dir, f'**/*.{file_ext[self.args.language]}'),
                                  recursive=True)

        if len(src_files) == 0:
            return project_context

        for filename in src_files:
            if os.path.exists(filename):  # weird but some files cannot be opened to read
                if os.path.isfile(filename):
                    try:
                        with open(filename, "r") as file:
                            file_content = file.read()
                    except:
                        with open(filename, "rb") as file:
                            file_content = file.read().decode(errors='replace')

                    fileid = os.path.relpath(filename, root_dir)
                    project_context[fileid] = file_content
            else:
                pass

        return project_context


def get_cfc(example, args, semantic_ranker, repositories):
    project_context = repositories[example["file_meta"]["repo_path"]]

    # def_ground_truths = {d['file_path']: d['range'] for d in example['definitions']}
    # ref_ground_truths = {r['file_path']: r['range'] for r in example['history_callees']}
    def_ground_truths = {}
    for d in example['definitions']:
        if d['file_path'] not in def_ground_truths:
            def_ground_truths[d['file_path']] = []
        def_ground_truths[d['file_path']].append(d['range'])

    ref_ground_truths = {}
    for d in example['history_callees']:
        if d['file_path'] not in ref_ground_truths:
            ref_ground_truths[d['file_path']] = []
        ref_ground_truths[d['file_path']].append(d['range'])

    status = None

    if len(project_context) == 0:
        example["cross_file_context"] = ""
        status = "project_not_found"
    else:
        rel_file_path = os.path.relpath(example["file_meta"]["file_path"],
                                        example["file_meta"]["repo_path"])

        code_files = find_files_within_distance_k(
            rel_file_path,
            list(project_context.keys()),
            k=args.crossfile_distance,
        )

        code_files = code_files[:args.maximum_cross_files]

        if args.force_include_gt_cf:  # Used to compute the ranking scores, force gt cf occurred in candidates.
            gt_cross_file_paths = set(list(def_ground_truths.keys()) + list(ref_ground_truths.keys()))
            not_in_code_files = []
            for gt_cross_file_path in gt_cross_file_paths:
                if gt_cross_file_path not in code_files:
                    not_in_code_files.append(gt_cross_file_path)

            j = 0
            for not_in_code_file in not_in_code_files:
                while j > -len(code_files):
                    j = j - 1
                    if code_files[j] not in gt_cross_file_paths:
                        code_files[j] = not_in_code_file
                        break

        code_chunks = []
        code_chunk_ids = []
        chunk_id2status = {}  # {0: no_def_no_ref, 1: has_def, 2: has_ref, 3: both}

        for code_file in code_files:
            lines = project_context[code_file].split("\n")
            line_status = np.array([CODE_BLOCK_STATUS["NO_DEF_NO_REF"]] * len(lines))

            if code_file in def_ground_truths.keys():
                # Only set the code block contains the method definition name as the gt.
                for line_range in def_ground_truths[code_file]:
                    line_status[line_range[0]: line_range[0]+1] += CODE_BLOCK_STATUS["HAS_DEF"]

            if code_file in ref_ground_truths.keys():
                for line_range in ref_ground_truths[code_file]:
                    line_status[line_range[0]: line_range[1]] += CODE_BLOCK_STATUS["HAS_REF"]

            # removing empty lines
            filtered_lines = []
            filtered_line_status = []

            for line, status in zip(lines, line_status):
                if line.strip():
                    filtered_lines.append(line)
                    filtered_line_status.append(status)

            c_id = 0
            for i in range(0, len(filtered_lines), SLIDING_WINDOW_SIZE):
                c = "\n".join(filtered_lines[i:i + CHUNK_SIZE])
                in_block_line_status = filtered_line_status[i: i + CHUNK_SIZE]

                if len(in_block_line_status) == 0:
                    continue

                tokenized_c = tokenize_nltk(c)
                if len(tokenized_c) > 0:
                    if args.mask_strategy != 'None':
                        if CODE_BLOCK_STATUS["BOTH"] in in_block_line_status:
                            continue

                    if args.mask_strategy == 'mask_def':
                        if CODE_BLOCK_STATUS["HAS_DEF"] in in_block_line_status:
                            continue

                    if args.mask_strategy == 'mask_ref':
                        if CODE_BLOCK_STATUS["HAS_REF"] in in_block_line_status:
                            continue

                    code_chunks.append(c)
                    code_chunk_ids.append(f"{code_file}|{c_id}")

                    if sum(in_block_line_status) != 0:
                        block_status = {
                            'def': [],
                            'ref': []
                        }

                        for index, s in enumerate(in_block_line_status):
                            if s == CODE_BLOCK_STATUS["HAS_DEF"]:
                                block_status['def'].append(index)
                            elif s == CODE_BLOCK_STATUS["HAS_REF"]:
                                block_status['ref'].append(index)
                            elif s == CODE_BLOCK_STATUS["BOTH"]:
                                # In the case of reference itself in its definition.
                                block_status['def'].append(index)
                                block_status['ref'].append(index)

                        chunk_id2status[f"{code_file}|{c_id}"] = block_status

                    c_id += 1

        clipped_code_chunks = code_chunks[:args.maximum_chunk_to_rerank]
        clipped_code_chunk_ids = code_chunk_ids[:args.maximum_chunk_to_rerank]

        if args.force_include_gt_cf:
            not_in_gt_code_chunk_ids = set()
            for chunk_id in chunk_id2status.keys():
                if chunk_id not in clipped_code_chunk_ids:

                    not_in_gt_code_chunk_ids.add(chunk_id)
                    # If exists code_file|c_id - 1, also includes it.
                    file_name, c_id = chunk_id.rsplit("|", 1)
                    prev_chunk_id = f"{file_name}|{int(c_id) - 1}"

                    if prev_chunk_id in code_chunk_ids and prev_chunk_id not in clipped_code_chunk_ids:
                        not_in_gt_code_chunk_ids.add(prev_chunk_id)

            j = 0
            for gt_code_chunk_id in not_in_gt_code_chunk_ids:
                while j > -len(clipped_code_chunk_ids):
                    j = j - 1
                    if clipped_code_chunk_ids[j] not in chunk_id2status.keys():
                        clipped_code_chunk_ids[j] = gt_code_chunk_id

                        origin_index = code_chunk_ids.index(gt_code_chunk_id)
                        clipped_code_chunks[j] = code_chunks[origin_index]
                        break

        if len(code_chunks) == 0:
            example["cross_file_context"] = {}
            status = "no_cross_file_context"

        else:
            cfc, meta_data = get_cross_file_context_from_chunks(
                args=args,
                prompt=example["prompt"],
                chunks=clipped_code_chunks,
                chunk_ids=clipped_code_chunk_ids,
                chunk_id2status=chunk_id2status,
                semantic_ranker=semantic_ranker
            )
            example["crossfile_context"] = {}
            example["crossfile_context"]["list"] = cfc

            status = meta_data

    return example, status


def reorder_candidates_with_strategy(candidate_chunks, candidate_ids, strategy, ranking_scores, query=None):
    assert len(candidate_chunks) == len(candidate_ids)

    half_size = int(QUERY_LENGTH / 2)
    if strategy == 'semitone':
        assert query is not None
        query_lines = query.split('\n')
        half_query = "\n".join(query_lines[-half_size:])

    if strategy == 'current_chunk':
        return candidate_chunks, candidate_ids, ranking_scores

    id2idx = dict()
    for j, cci in enumerate(candidate_ids):
        id2idx[cci] = j

    reordered_candidate_chunks = []
    reordered_candidate_ids = []
    reordered_ranking_scores = []

    
    left_chunks = []
    left_chunk_ids = []
    left_scores = []

    for candidate_id, candidate_chunk in zip(candidate_ids, candidate_chunks):
        file_name, c_id = candidate_id.rsplit("|", 1)
        next_id = f"{file_name}|{int(c_id) + 1}"

        if strategy == 'next_chunk':
            if next_id in candidate_ids and next_id not in reordered_candidate_ids:
                reordered_candidate_ids.append(next_id)
                reordered_candidate_chunks.append(candidate_chunks[id2idx[next_id]])
                if ranking_scores is not None:
                    reordered_ranking_scores.append(ranking_scores[id2idx[next_id]])

                left_chunk_ids.append(candidate_id)
                left_chunks.append(candidate_chunk)
                if ranking_scores is not None:
                    left_scores.append(ranking_scores[id2idx[candidate_id]])

            elif candidate_id not in reordered_candidate_ids:
                reordered_candidate_ids.append(candidate_id)
                reordered_candidate_chunks.append(candidate_chunk)
                if ranking_scores is not None:
                    reordered_ranking_scores.append(ranking_scores[id2idx[candidate_id]])

        elif strategy == 'current_and_next_chunk':
            if candidate_id not in reordered_candidate_ids:
                reordered_candidate_ids.append(candidate_id)
                reordered_candidate_chunks.append(candidate_chunk)
                if ranking_scores is not None:
                    reordered_ranking_scores.append(ranking_scores[id2idx[candidate_id]])

            if next_id in candidate_ids and next_id not in reordered_candidate_ids:
                reordered_candidate_ids.append(next_id)
                reordered_candidate_chunks.append(candidate_chunks[id2idx[next_id]])
                if ranking_scores is not None:
                    reordered_ranking_scores.append(ranking_scores[id2idx[next_id]])

    for j in range(len(left_chunk_ids)):
        if left_chunk_ids[j] not in reordered_candidate_ids:
            reordered_candidate_ids.append(left_chunk_ids[j])
            reordered_candidate_chunks.append(left_chunks[j])
            if ranking_scores is not None:
                reordered_ranking_scores.append(left_scores[j])

    assert len(reordered_candidate_ids) == len(candidate_ids)

    if ranking_scores is None:
        return reordered_candidate_chunks, reordered_candidate_ids, None
    else:
        return reordered_candidate_chunks, reordered_candidate_ids, reordered_ranking_scores


def swap_item_by_id(chunk_list, index_a, index_b):
    if chunk_list is None:
        return

    item_a = chunk_list[index_a]
    item_b = chunk_list[index_b]

    chunk_list[index_a] = item_b
    chunk_list[index_b] = item_a


def get_cross_file_context_from_chunks(args,
                                       prompt,
                                       chunks,
                                       chunk_ids,
                                       chunk_id2status,
                                       semantic_ranker):
    assert len(chunks) != 0

    ranking_scores = None
    meta_data = {}

    prompt_lines = [pl for pl in prompt.split("\n") if pl.strip()]
    query = "\n".join(prompt_lines[-QUERY_LENGTH:])

    meta_data["query"] = query

    if args.rerank:
        start = time.time()

        if args.ranking_fn == "cosine_sim":
            gpu_id = int(mp.current_process().name.split('-')[-1]) - 1
            chunks, chunk_ids, ranking_scores = semantic_ranker.rerank(
                query,
                chunks,
                chunk_ids,
                gpu_id,
                score_threshold=None
            )
        else:
            chunks, chunk_ids, ranking_scores = lexical_ranking(
                query,
                chunks,
                args.ranking_fn,
                chunk_ids,
                score_threshold=None
            )

        meta_data["latency"] = time.time() - start
        meta_data["num_candidates"] = len(chunks)

    top_k = min(args.maximum_cross_file_chunk, len(chunks))
    if top_k == 0:
        return [], meta_data

    chunks, chunk_ids, ranking_scores = reorder_candidates_with_strategy(
        chunks,
        chunk_ids,
        args.concat_strategy,
        ranking_scores,
        query
    )

    rerank_ids = []

    if args.noise_ratio != -1 or args.distract_ratio != -1:
        # print("Building generation dataset with noise (RQ2).")
        gt_num = 1
        if args.noise_ratio != -1:
            noise_num = gt_num * args.noise_ratio
        if args.distract_ratio != -1:
            noise_num = gt_num * args.distract_ratio

        if noise_num + gt_num > top_k:
            noise_num = top_k
            gt_num = 0

        top_k = gt_num + noise_num

        # Kept gts
        gt_indexes = sorted([chunk_ids.index(k) for k in chunk_id2status.keys()])
        assert len(gt_indexes) > 0
        selected_gt_indexes = gt_indexes[:gt_num]

        # Select similar but not unrelated noises.
        related_noise_indexes = []
        if args.distract_ratio != -1:
            for j, chunk_id in enumerate(chunk_ids):
                if len(related_noise_indexes) < noise_num:
                    if chunk_id not in chunk_id2status:
                        related_noise_indexes.append(j)
                else:
                    break

        if args.noise_ratio != -1:
            random_start = min(3 * top_k, len(chunk_ids) - noise_num)
            noise_indexes = [j + random_start for j, _id in enumerate(chunk_ids[random_start:]) if _id not in chunk_id2status]
            if noise_num > 0:
                related_noise_indexes = random.sample(noise_indexes, noise_num)

        rerank_ids = selected_gt_indexes + related_noise_indexes

    if args.test_reason_ability != "None":
        # "None", "ref_only", "def_only", "def_and_ref"
        ref_gt_indexes = sorted([chunk_ids.index(k) for k, v in chunk_id2status.items() if len(v['ref']) > 0])
        def_gt_indexes = sorted([chunk_ids.index(k) for k, v in chunk_id2status.items() if len(v['def']) > 0])

        if len(def_gt_indexes) == 0 or len(ref_gt_indexes) == 0:
            return [], "not_both_ref_and_def"

        id_len = min(len(def_gt_indexes), len(ref_gt_indexes), top_k // 2)

        if args.test_reason_ability in ['def_only', 'def_and_ref']:
            rerank_ids += def_gt_indexes[: id_len]
        if args.test_reason_ability in ['ref_only', 'def_and_ref']:
            rerank_ids += ref_gt_indexes[: id_len]

        if len(rerank_ids) < top_k:  # Only with gt.
            top_k = len(rerank_ids)

    if chunk_id2status is not None:
        ref_gts = set()
        def_gts = set()
        both_gts = set()

        for k, v in chunk_id2status.items():
            both_gts.add(k)

            if len(v['ref']) > 0:
                ref_gts.add(k)

            if len(v['def']) > 0:
                def_gts.add(k)

        ref_scores = None
        if len(ref_gts) > 0:
            ref_scores = evaluate_retrieval_metric(chunk_ids, ref_gts,
                                                   predict_scores=ranking_scores, top_k=top_k)
        def_scores = None
        if len(def_gts) > 0:
            def_scores = evaluate_retrieval_metric(chunk_ids, def_gts,
                                                   predict_scores=ranking_scores, top_k=top_k)

        both_scores = None
        if len(ref_gts) > 0 or len(def_gts) > 0:
            both_scores = evaluate_retrieval_metric(chunk_ids, both_gts,
                                                    predict_scores=ranking_scores, top_k=top_k)

        meta_data['scores'] = {
            'ref': ref_scores,
            'def': def_scores,
            'both': both_scores,
        }

    selected_chunks_scores = []

    if rerank_ids is not None:
        if args.rerank:
            selected_chunks_scores = [ranking_scores[j] for j in rerank_ids]

            rerank_ids = [_id for _id, score in sorted(zip(rerank_ids, selected_chunks_scores),
                                                       key=lambda x: x[1], reverse=True)]
            selected_chunks_scores = [ranking_scores[j] for j in rerank_ids]

        selected_chunks = [chunks[j] for j in rerank_ids]
        selected_chunk_ids = [chunk_ids[j] for j in rerank_ids]
        selected_chunks_filename = [chunk_ids[j].rsplit("|", 1)[0] for j in rerank_ids]

    else:
        selected_chunks = chunks[:top_k]
        selected_chunk_ids = chunk_ids[:top_k]
        selected_chunks_filename = [_id.rsplit("|", 1)[0] for _id in chunk_ids[:top_k]]
        if args.rerank:
            selected_chunks_scores = ranking_scores[:top_k]

    cross_file_context = []
    for idx in range(len(selected_chunks)):
        retrieval_content = {
            "retrieved_chunk": selected_chunks[idx],
            "filename": selected_chunks_filename[idx],
            "score": selected_chunks_scores[idx] if args.rerank else None,
        }
        if chunk_id2status is not None:
            select_chunk_id = selected_chunk_ids[idx]
            if select_chunk_id in chunk_id2status.keys():
                retrieval_content['hit'] = chunk_id2status[select_chunk_id]
            else:
                retrieval_content['hit'] = -1

        cross_file_context.append(retrieval_content)

    # Varify
    if args.test_reason_ability != "None":
        if args.test_reason_ability in ['def_only', 'def_and_ref']:
            for r in cross_file_context:
                if isinstance(r['hit'], int):
                    raise Exception("Only GT should occur in prompt.")

    if args.noise_ratio != -1 or args.distract_ratio != -1:
        error_num = 0
        for r in cross_file_context:
            if isinstance(r['hit'], int):
                error_num += 1
        if error_num != noise_num:
            raise Exception(f"Noise Exception! Expect {noise_num} noises in prompt,"
                            f"actual {error_num} noises in prompt.")

    return cross_file_context, meta_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rerank",
        type=str2bool,
        default=False,
        help="rerank the functions"
    )
    parser.add_argument(
        "--ranker",
        type=str,
        default="sparse",
        choices=["sparse", "unixcoder"],
        help="ranking function"
    )
    parser.add_argument(
        "--ranking_fn",
        type=str,
        default="path_distance",
        choices=["path_distance", "tfidf", "bm25", "jaccard_sim", "cosine_sim"],
        help="ranking function"
    )

    parser.add_argument(
        "--force_include_gt_cf",
        type=str2bool,
        default=False,
        help="Force including the ground truth of cross-file context in the retrieval candidates,"
             " used for computing MAP, MRR and Pass@k metrics."
    )

    parser.add_argument(
        '--noise_ratio',
        type=int,
        default=-1,
        help="The ratio of noise, if -1, do nothing. noise_ratio = [0, 1, 2, 3, 4, 5, 6] "
             "noise_num = (truth_num * noise_ratio)"
             "if 0, only gts in prompt; if 6, only noise in prompt."
    )

    parser.add_argument(
        '--distract_ratio',
        type=int,
        default=-1,
        help="The ratio of distract, if -1, do nothing."
    )

    parser.add_argument(
        "--test_reason_ability",
        type=str,
        default="None",
        choices=["None", "ref_only", "def_only", "def_and_ref"],
        help="test whether the LLM can reason from [def, ref, def and ref]. Only test samples"
             "with both ref gts and def gt are selected for this experiment."
    )

    parser.add_argument(
        "--crossfile_distance",
        type=int,
        default=100,
        help="max distance to search for crossfile"
    )
    parser.add_argument(
        "--maximum_chunk_to_rerank",
        type=int,
        default=1000,
        help="max chunks to consider to rank via BM25"
    )
    parser.add_argument(
        "--maximum_cross_files",
        type=int,
        default=1000,
        help="max chunks to consider to rank via BM25"
    )
    parser.add_argument(
        "--maximum_cross_file_chunk",
        type=int,
        default=50,
        help="max chunks to return as cfc"
    )
    parser.add_argument(
        "--concat_strategy",
        type=str,
        default="current_chunk",
        help="[current_chunk, next_chunk, current_and_next_chunk, semitone]"
    )

    parser.add_argument(
        '--mask_strategy',
        type=str,
        default="None",
        help="[mask_def, mask_ref]"
    )

    parser.add_argument(
        '--sub_dataset',
        type=str2bool,
        default=False,
        help="Only using samples with both ref and def gts."
    )

    parser.add_argument(
        "--skip_if_no_cfc",
        type=str2bool,
        default=True,
        help="skip adding examples if there is no crossfile context"
    )
    parser.add_argument(
        "--output_file_suffix",
        type=str,
        default=None,
        help="add a suffix string to the output file"
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        choices=["Java", "Python", "TypeScript", "JavaScript"],
        help="language name"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="The line-completion file."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="The root dir of repos."
    )
    parser.add_argument(
        "--use_codalab",
        type=str2bool,
        default=True,
        help="If use codalab, output files should upload to ftp server."
    )

    args = parser.parse_args()

    args.output_file_suffix = "" if args.output_file_suffix is None else f"_{args.output_file_suffix}"

    tgtfile_suffix = ""

    if args.rerank:
        tgtfile_suffix += f"_{args.ranking_fn}_{args.concat_strategy}"
    else:
        tgtfile_suffix += f"_rank_by_path_{args.concat_strategy}"

    if args.noise_ratio != -1:
        tgtfile_suffix += f"_noise_{args.noise_ratio}"

    if args.distract_ratio != -1:
        tgtfile_suffix += f"_distract_{args.distract_ratio}"

    if args.sub_dataset:
        tgtfile_suffix += f"_only_subset"

    if args.test_reason_ability != "None":
        tgtfile_suffix += f"_reason_ability_{args.test_reason_ability}"

    if args.mask_strategy != 'None':
        tgtfile_suffix += f"_{args.mask_strategy}"

    args.num_processes = 10
    if args.ranking_fn == "cosine_sim":
        num_gpus = torch.cuda.device_count()
        args.num_processes = num_gpus
        mp.set_start_method('spawn')

    test_samples = []
    i = 0

    with jsonlines.open(args.input_file) as f:
        for line in f:
            line['idx'] = i
            i += 1

            if args.sub_dataset:
                if len(line['definitions']) > 0 and len(line['history_callees']) > 0:
                    test_samples.append(line)
            else:
                test_samples.append(line)

    output_path = os.path.dirname(args.input_file)
    output_filename = os.path.splitext(os.path.basename(args.input_file))[0]
    output_filename = output_filename + args.output_file_suffix + tgtfile_suffix + ".jsonl"

    r = CodeRetrieval(args, repository_root=args.root_dir, test_samples=test_samples, use_codalab=args.use_codalab)
    r.retrieve(output_path, output_filename)
