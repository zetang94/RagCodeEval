import argparse
import logging
from collections import Counter
from itertools import groupby
from pathlib import Path
import re
import random

import git
import os
from shutil import copytree
from jsonlines import jsonlines
import pandas as pd
from tqdm import tqdm
from call_parser import langs
import json

from vo import FileMeta, DefMeta, CalleeMeta, Sample
from stack_graph_helper import StackGraphHelper
from utils import split_dataset, analyze_ref_def_in_repo, get_commits_by_path, extract_funcs_from_file
from utils import extract_funcs_from_repo


logger = logging.getLogger("build_dataset")


def set_logger_file(lang):
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f"logs/{lang}_build_dataset.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

    # 将处理程序添加到logger对象
    logger.addHandler(console_handler)
    logger.addHandler(fh)


def remove_prefix(text, prefix):
    if text and text.startswith(prefix):
        return text[len(prefix):]
    return text


def list_to_str(source_list):
    if isinstance(source_list, list):
        return "\n".join(source_list)
    return source_list


# def save_to_csv(dataframe, repo_dir, csv_path):
#     df_copy = dataframe.copy()
#     if 'Callees' in csv_path or 'Definitions' in csv_path:
#         df_copy['signature'] = df_copy['signature'].apply(remove_prefix, args=(repo_dir,))
#         df_copy['file_path'] = df_copy['file_path'].apply(remove_prefix, args=(repo_dir,))
#     elif 'Functions' in csv_path:
#         df_copy['file_path'] = df_copy['file_path'].apply(remove_prefix, args=(repo_dir,))
#         df_copy.drop(['callee'], axis=1)
#     elif 'Files' in csv_path:
#         df_copy['file_path'] = df_copy['file_path'].apply(remove_prefix, args=(repo_dir,))
#         df_copy['source_code'] = df_copy['source_code'].apply(list_to_str)
#
#     df_copy.to_csv(csv_path)


# def post_process_call(ref_info_list, repo_dir):
#     for ref_info in ref_info_list:
#         if ref_info and 'file_path' in ref_info:
#             ref_info['file_path'] = remove_prefix(ref_info['file_path'], repo_dir)
#         if ref_info and 'signature' in ref_info:
#             ref_info['signature'] = remove_prefix(ref_info['signature'], repo_dir)
#
#     return ref_info_list


def filter_with_lang_and_sort_by_stars(process_lang):
    commit_meta_path = "metadata/repo_commit_metadata.jsonl"
    commit_metadata = pd.read_json(commit_meta_path, orient="records", lines=True)

    filtered_df = pd.read_json("./metadata/filtered_ghs_results_05_mar_2023.jsonl", orient="records", lines=True)
    filtered_df = filtered_df.merge(commit_metadata, on=["name", 'mainLanguage'], how="inner")

    filtered_df = filtered_df.loc[(~filtered_df.local_path.isna()) & (filtered_df['mainLanguage'] == process_lang)]

    filtered_df.sort_values(by=['stargazers'], ascending=True, ignore_index=True)

    return filtered_df


def calculate_processed_repos(process_lang):
    process_repos = set()
    num_samples = 0
    with jsonlines.open(f"../data/dataset/{process_lang}/line_completion.jsonl") as f:
        for line in f:
            process_repos.add(line['file_meta']['repo_name'])
            num_samples += 1

    return len(process_repos), num_samples


def skip_repo(call_parser, added_methods, commit_file_path):
    filtered_added_methods = []

    with open(commit_file_path + "commit_statistic.jsonl", 'r') as f:
        commit_statistic = json.load(f)

    cur_contents = None
    current_commit_chunk_path = None
    previous_commit_id = None
    start = False
    for i, added_method in enumerate(added_methods):
        commit_id = added_method['commitId']

        if commit_id == '9e04c34e2070b7e615e3be276f453232d554527d':
            if not start:
                print("Method Start ", i)
            start = True
        else:
            previous_commit_id = commit_id
            if start:
                print("Previous Commit ID", previous_commit_id)
                print("Method END ", i)
                break


        method_range = added_method['pos']
        file_path = added_method['filePath']

        if cur_contents is None or commit_statistic[commit_id] != current_commit_chunk_path:
            current_commit_chunk_path = commit_statistic[commit_id]
            cur_contents = get_commits_by_path(current_commit_chunk_path)

        source_code = cur_contents[commit_id][file_path]
        # source_code = source_code.split('\n')
        # method_body = source_code[method_range[0]: method_range[1]]
        callees, _ = call_parser.extract(source_code)
        callee_lines = [callee[3] for callee in callees if method_range[0]< callee[3] < method_range[1]]

        if len(callee_lines) > 0:
            filtered_added_methods.append(added_method)

    commit_id_list = []
    for add_method in filtered_added_methods:
        commit_id_list.append(add_method['commitId'])

    commit_counter = Counter(commit_id_list)
    most_common_commit_dict = commit_counter.most_common(1)[0]
    if most_common_commit_dict[1] / len(commit_id_list) > 0.5:
        return True
    else:
        return False


def build_dataset(prefix_path, process_lang, stack_graph_path):
    tmp_path = prefix_path + "snapshot/"

    filtered_df = filter_with_lang_and_sort_by_stars(process_lang)

    total_repo_num = len(filtered_df)

    lifecycles = ['Initiation', 'Intermediate', 'Closure']
    split_num = len(lifecycles)

    processed_repo_num = 0
    sample_num = 0

    used_repo_num = 0

    test_sample_path = f"../data/dataset/{process_lang}"

    if not os.path.exists(test_sample_path):
        Path(test_sample_path).mkdir(parents=True, exist_ok=True)
    else:
        used_repo_num, sample_num = calculate_processed_repos(process_lang)
        logger.info(f"Restart from the processed snapshot, generate {used_repo_num}/{sample_num} samples.")

    for row in filtered_df.itertuples():

        if used_repo_num >= 100 and sample_num >= 1000:
            continue

        repo_ground_truths = []
        repo_samples = [[], [], []]

        name = getattr(row, 'name')
        lang = getattr(row, 'mainLanguage')
        local_path = getattr(row, 'local_path')
        commit_path = getattr(row, 'commit_path')
        commit_file_path = getattr(row, 'commit_file_path')

        if lang != process_lang and process_lang != "all":
            continue

        processed_repo_num += 1
        # Parser
        call_parser = langs[lang]
        # stack-graph
        stack_graph = StackGraphHelper(lang, path=stack_graph_path)

        # Read commits
        raw_samples = open(commit_path, "r")
        added_methods = [sample for sample in list(map(json.loads, raw_samples))]
        added_methods.sort(key=lambda method: method["commitTime"])

        logger.info(f"Process {processed_repo_num}/{total_repo_num} with name {name} and method num {len(added_methods)}")

        split_commits = split_dataset(added_methods, split_num + 1)

        if os.path.exists(tmp_path + lang + "/" + name.replace("/", "@")):
            continue

        if split_commits is None:
            logger.info(
                f"    Repo {name} is skipped, because no more than 50 methods or cannot find suitable split snapshot id.")
            continue

        split_commits = split_commits[:split_num]

        for i, split_commit in enumerate(split_commits):
            tmp_repo_path = tmp_path + lang + "/" + name.replace("/", "@") + "/" + lifecycles[i] + "/"

            total_commit_ids = set()
            for m in added_methods[split_commit['method_range'][0]: split_commit['method_range'][1]]:
                total_commit_ids.add(m["commitId"])

            start_commit = split_commit['start_commit']

            # copy the start_commit snapshot to the tmp dir
            repo = git.Repo(local_path)
            repo.head.reset(start_commit, index=True, working_tree=True)

            logger.info(f"    Repo {name}, {lifecycles[i]}.   STEP 1: copy repo to tmp dir.")
            copytree(local_path, tmp_repo_path, symlinks=True)

            logger.info(f"    Repo {name}, {lifecycles[i]}.   STEP 2: extract calls from tmp dir.")
            repo_func_list, file_info_list = extract_funcs_from_repo(tmp_repo_path, call_parser)

            # To avoid too much retrieval or too little retrieval.
            if len(repo_func_list) > 10000 or len(repo_func_list) < 50:
                break

            repo_func_df = pd.DataFrame.from_records(repo_func_list)

            logger.info(f"    Repo {name}, {lifecycles[i]}.   STEP 3: stack graph init and index.")
            stack_graph.clean()
            stack_graph.set_current_repo_path(tmp_repo_path, call_parser.extension)
            skip_file_paths = stack_graph.index_repo(tmp_repo_path)
            if skip_file_paths is None:
                logger.info(f"    Repo {name}, {lifecycles[i]}.   STEP 3: Index out of time, continue.")
                break

            logger.info(f"    Repo {name}, {lifecycles[i]}.   "
                        f"STEP 3: index {len(file_info_list) - len(skip_file_paths)}/{len(file_info_list)} files. ")

            logger.info(f"    Repo {name}, {lifecycles[i]}.   STEP 4: find ref-def pairs from stack.")
            ref_def_result = analyze_ref_def_in_repo(stack_graph,
                                                     repo_func_df,
                                                     repo_func_df,
                                                     skip_file_paths=skip_file_paths,
                                                     is_train=True)

            repo_callee_list = ref_def_result['callee_list']
            repo_called_edges = ref_def_result['called_edges']
            repo_def_sig_index_map = ref_def_result['def_sig_index_map']

            # 5.按照[commit_id, commit_time, file_path, method_list]来将要处理的method进行重新组织
            cur_range = split_commit['method_range']
            cur_contents = None
            current_commit_chunk_path = ""
            with open(commit_file_path + "commit_statistic.jsonl", 'r') as f:
                commit_statistic = json.load(f)

            processed_commit_ids = set()

            logger.info(f"    Repo {name}, {lifecycles[i]}.  "
                        f" STEP 6: Generate sample from {cur_range[1] - cur_range[0]} later commit methods.")

            commit_bar = tqdm(total=len(total_commit_ids))

            for key, group in groupby(added_methods[cur_range[0]: cur_range[1]],
                                      lambda x: (x['commitId'], x['commitTime'], x['filePath'])):
                include_method_names = [m['methodName'] for m in group]

                commit_id, commit_time, file_path = key
                bar_update_n = 1 if commit_id not in processed_commit_ids else 0
                processed_commit_ids.add(commit_id)

                if cur_contents is None or commit_statistic[commit_id] != current_commit_chunk_path:
                    current_commit_chunk_path = commit_statistic[commit_id]
                    cur_contents = get_commits_by_path(current_commit_chunk_path)

                source_code = cur_contents[commit_id][file_path]
                source_code_before = None
                tmp_commit_file_path = tmp_repo_path + file_path
                # 保留之前的文件信息
                if os.path.exists(tmp_commit_file_path):
                    try:
                        with open(tmp_commit_file_path, 'r', encoding='utf-8') as f:
                            source_code_before = f.read()
                    except Exception as e:
                        source_code_before = None
                # 替换文件
                tmp_dir_path = os.path.dirname(tmp_commit_file_path)
                if not os.path.exists(tmp_dir_path):
                    Path(tmp_dir_path).mkdir(parents=True, exist_ok=True)
                with open(tmp_commit_file_path, 'w') as f:
                    f.write(source_code)
                stack_graph.index_repo(tmp_repo_path)
                extract_result = extract_funcs_from_file(tmp_repo_path, tmp_commit_file_path, call_parser)
                if extract_result is None:
                    continue
                file_func_list, current_file_info = extract_result
                file_func_pd = pd.DataFrame.from_records(file_func_list)

                file_ref_def_result = analyze_ref_def_in_repo(stack_graph,
                                                              repo_func_df,
                                                              file_func_pd,
                                                              skip_file_paths=skip_file_paths,
                                                              is_train=False)

                lines = current_file_info['source_code']

                file_callee_list = file_ref_def_result['callee_list']
                file_def_list = file_ref_def_result['def_list']
                file_callee_edges = file_ref_def_result['callee_edges']
                file_called_edges = file_ref_def_result['called_edges']

                # 找到历史调用这个方法的代码
                for callee_index in file_callee_edges.keys():
                    current_callee_info = file_callee_list[callee_index]
                    if current_callee_info['parent_scope'] not in include_method_names:
                        continue
                    current_callee_line_no = current_callee_info['line_no']

                    # extract history cross file calls.
                    history_callee_sig_set = {'out_file': set(), 'in_file': set()}
                    history_def_sig_set = {'out_file': set(), 'in_file': set()}
                    history_cross_file_callees = []
                    history_in_file_callees = []

                    in_file_definitions = []
                    out_file_definitions = []

                    deleted_import_nos = []
                    deleted_import_stm = []

                    for called_def_index, ref_cross_file_def in file_callee_edges[callee_index]:

                        called_def_info = file_def_list[called_def_index]
                        called_def_type = called_def_info['def_type']
                        called_def_sig = called_def_info['signature']

                        if called_def_type in ['class_declaration', 'method_declaration']:
                            if ref_cross_file_def:
                                if called_def_sig not in history_def_sig_set['out_file']:
                                    history_def_sig_set['out_file'].add(called_def_sig)
                                    out_file_definitions.append(called_def_info)
                            else:
                                if called_def_sig not in history_def_sig_set['in_file']:
                                    history_def_sig_set['in_file'].add(called_def_sig)
                                    in_file_definitions.append(called_def_info)

                        elif called_def_type == 'import_declaration':
                            if not ref_cross_file_def:
                                import_line_no = called_def_info['range'][0]
                                if import_line_no < current_callee_line_no:
                                    if import_line_no not in deleted_import_nos:
                                        deleted_import_nos.append(import_line_no)
                                        deleted_import_stm.append(lines[import_line_no])

                        # Deal with in-file history callees.
                        for _id, _ in file_called_edges[called_def_index]:
                            history_callee_info = file_callee_list[_id]
                            if history_callee_info['line_no'] < current_callee_line_no:
                                callee_sig = history_callee_info['signature']

                                if callee_sig not in history_callee_sig_set['in_file']:
                                    history_callee_sig_set['in_file'].add(callee_sig)
                                    history_in_file_callees.append(history_callee_info)

                        # Deal with cross-file history callees.
                        if called_def_sig in repo_def_sig_index_map:
                            repo_def_index = repo_def_sig_index_map[called_def_sig]
                            # repo_def_info = repo_def_list[repo_def_index]

                            for _id, _ in repo_called_edges[repo_def_index]:
                                history_callee_info = repo_callee_list[_id]
                                if history_callee_info['file_path'] != file_path:
                                    callee_sig = history_callee_info['signature']

                                    if callee_sig not in history_callee_sig_set['out_file']:
                                        history_callee_sig_set['out_file'].add(callee_sig)
                                        history_cross_file_callees.append(history_callee_info)

                    prompt = [code_line for j, code_line in enumerate(
                        lines) if j < current_callee_line_no and j not in deleted_import_nos]
                    ground_truth = lines[current_callee_line_no] + "\n"

                    match = re.search(r'\S', ground_truth)
                    if match:
                        start_index = match.start()
                        prompt.append(ground_truth[:start_index])
                        ground_truth = ground_truth[start_index:]

                    prompt = "\n".join(prompt)
                    right_context = "\n".join(lines[current_callee_line_no + 1:])

                    if len(history_in_file_callees) > 0 or len(in_file_definitions) > 0:
                        # Current file contains similar calls/definitions, not needs to retrieval
                        continue

                    no_cross_callees = len(history_cross_file_callees) == 0
                    no_cross_def = len(out_file_definitions) == 0

                    if no_cross_callees and no_cross_def:
                        # Non cross file callees or cross file definitions, no need to retrieval.
                        # Currently, stack-graph-python has bugs, so we should manually find the definition
                        # of imported class.
                        continue

                    # To avoid the sample contains too long strings.
                    if len(ground_truth) > 256:
                        #logger.info("    Sample is skipped because of too long ground truth(>256).")
                        continue

                    # To avoid too frequent call names occur many times in the test set,
                    # We use called_name + has_cross_callee + has_cross_def to avoid this.
                    sample_sig = current_callee_info['called_func_name'] + ":" + str(no_cross_callees) + ":" + str(
                        no_cross_def
                    )

                    if sample_sig in repo_ground_truths:
                        #logger.info(f"    Sample is skipped because of existed called func name {sample_sig}.")
                        continue
                    else:
                        repo_ground_truths.append(sample_sig)

                    #post_process_call(out_file_definitions, prefix_path)
                    #post_process_call(history_cross_file_callees, prefix_path)

                    file_meta = FileMeta(repo_name=name,
                                         file_path=file_path,
                                         commit_id=commit_id,
                                         lifecycle=lifecycles[i])

                    def_meta = []

                    for definition in out_file_definitions:
                        def_meta.append(
                            DefMeta(file_path=definition['file_path'],
                                    def_stmt=definition['def_stmt'],
                                    def_type=definition['def_type'],
                                    def_range=definition['range'])
                        )

                    history_callees = []

                    for cross_callee in history_cross_file_callees:
                        history_callees.append(
                            CalleeMeta(called_func_name=cross_callee['called_func_name'],
                                       file_path=cross_callee['file_path'],
                                       callee_stmt=cross_callee['call_stmt'],
                                       callee_range=[cross_callee['line_no'],
                                                     cross_callee['line_no']+1]
                                       )
                        )

                    sample = Sample(
                        prompt=prompt,
                        ground_truth=ground_truth,
                        right_context=right_context,
                        delete_import_lines=deleted_import_stm,
                        delete_line_nos=deleted_import_nos,
                        history_callees=history_callees,
                        def_meta=def_meta,
                        file_meta=file_meta
                    )

                    repo_samples[i].append(sample.to_dict())

                    #print(sample.to_dict())

                # 替换文件
                if source_code_before is not None:
                    with open(tmp_commit_file_path, 'w') as f:
                        f.write(source_code_before)
                else:
                    assert os.path.exists(tmp_commit_file_path)
                    os.remove(tmp_commit_file_path)

                commit_bar.set_description(f"{len(processed_commit_ids)}/{len(total_commit_ids)} commits processed.")
                commit_bar.update(bar_update_n)

                # if len(test_samples) > sample_chunk:
                #     with jsonlines.open(test_sample_path + "/line_completion.jsonl", "a") as writer:
                #         writer.write_all(test_samples)
                #         test_samples = []

        # if test_samples:
        #     with jsonlines.open(test_sample_path + "/line_completion.jsonl", "a") as writer:
        #         writer.write_all(test_samples)

        wrapped_samples = []
        for lifecycle_samples in repo_samples:
            if len(lifecycle_samples) > 50:
                logger.info(f"    Keep 50 samples for repo {name}.")
                selected_samples = random.sample(lifecycle_samples, 50)
                wrapped_samples.extend(selected_samples)
            else:
                wrapped_samples.extend(lifecycle_samples)

        # Only one sample is skipped.
        if len(wrapped_samples) > 1:
            used_repo_num += 1
            with jsonlines.open(test_sample_path + "/line_completion.jsonl", "a") as writer:
                writer.write_all(wrapped_samples)
            sample_num += len(wrapped_samples)

        logger.info(f"Generate {len(wrapped_samples)} from repo {name}, current generate {used_repo_num}/{sample_num} samples.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="The absolute dataset path", type=str)
    parser.add_argument("-l", "--lang", help="Languages to process", default="all", type=str)
    parser.add_argument("-s", "--stack_graph_path", help="The stack graph path", type=str)

    args = parser.parse_args()
    set_logger_file(args.lang)

    build_dataset(args.path, args.lang, args.stack_graph_path)

