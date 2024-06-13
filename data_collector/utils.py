import os
from typing import Optional

from jsonlines import jsonlines
from pydriller import Repository, ModificationType
import pandas as pd
from tqdm import tqdm
import lizard
import json


def avg_split_list(arr, split_num):
    """
    :param arr: Integer list, and each element > 0
    :param split_num: The number of sub lists needs to be split.
    :return: [partition indexes], which can make the (split_num + 1) sub lists that have the close sum value.
    """
    n = len(arr)
    sub_list_num = split_num + 1
    prefix_sum = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix_sum[i] = prefix_sum[i - 1] + arr[i - 1]

    # dp[i][j] represents the minimum possible maximum subarray sum for the first i numbers using j partitions.
    dp = [[float('inf')] * sub_list_num for _ in range(n + 1)]
    partition = [[-1] * sub_list_num for _ in range(n + 1)]
    dp[0][0] = 0

    for i in range(1, n + 1):
        for j in range(1, min(sub_list_num, i + 1)):  # Now allowing up to 5 partitions, i.e., 6 segments
            for k in range(i):
                current_sum = prefix_sum[i] - prefix_sum[k]
                max_sum = max(dp[k][j - 1], current_sum)
                if dp[i][j] > max_sum:
                    dp[i][j] = max_sum
                    partition[i][j] = k

    # Backtracking to find the partition points
    cuts = []
    j = split_num  # Starting from partition into 5 segments
    i = n
    while j > 0:
        i = partition[i][j]
        if i > 0:  # To avoid appending 0, which is not a valid cut
            cuts.append(i)
        j -= 1

    cuts = sorted(cuts)
    return cuts


def split_dataset(added_methods,
                  split_index_num=4,) -> Optional[str]:
    commit_ids = []
    func_nums = []

    for i, added_method in enumerate(added_methods):
        current_id = added_method["commitId"]
        if i > 0 and commit_ids[-1][0] == current_id:
            func_nums[-1] += 1
        else:
            commit_ids.append((current_id, i))
            func_nums.append(1)

    results = []

    indexes = avg_split_list(func_nums, split_index_num)

    ## Notice, ignore repos with less than 50 methods.
    if len(indexes) != (split_index_num - 1) or len(added_methods) < 50:
        return None

    for i in range(len(indexes)):
        snapshot_id = commit_ids[indexes[i]-1][0]
        start_func_id = commit_ids[indexes[i]][1]
        if i > 0:
            results[-1]['method_range'].append(start_func_id)
            results[-1]['last_commit'] = snapshot_id
        results.append({
            'start_commit': snapshot_id,
            'method_range': [start_func_id,]
        })

    results[-1]['method_range'].append(len(added_methods))
    results[-1]['last_commit'] = commit_ids[-1][0]

    return results


def download_all_commits(repo_path, file_extension, bar=None, output_path=None):
    result = []

    modify_files = []

    code_list = []

    commit_statics = {}
    j = 0

    for commit in Repository(repo_path,
                             only_modifications_with_file_types=[file_extension],
                             ).traverse_commits():
        commit_id = commit.hash
        cur_commit_time = int(commit.committer_date.strftime("%Y%m%d%H%M%S"))

        if bar is not None:
            bar.set_description(
                f"Deal with {file_extension}: process repo {repo_path.split('/')[-1]} at commit {commit_id[:4]},"
                f" processed ")

        for modify_file in commit.modified_files:
            if not modify_file.filename.endswith(file_extension):
                continue

            if modify_file.change_type == ModificationType.ADD:
                add_methods = modify_file.methods
            elif modify_file.change_type == ModificationType.MODIFY:
                old_methods = modify_file.methods_before
                new_methods = modify_file.methods
                old_method_signatures = [m.long_name for m in old_methods]
                add_methods = []
                for new_method in new_methods:
                    if new_method.long_name not in old_method_signatures:
                        add_methods.append(new_method)
            else:
                # there is no new added methods in [DELETE or RENAME], so just continue.
                continue

            code = modify_file.source_code
            if code is None:
                continue

            clear_code = "".join(code.split())
            if clear_code not in code_list:
                code_list.append(clear_code)
            else:
                # Continue with
                continue

            for add_method in add_methods:
                method_sig = add_method.long_name

                result.append({
                    "methodName": method_sig,
                    "commitId": commit_id,
                    "commitTime": cur_commit_time,
                    "filePath": modify_file.new_path,
                    "fileName": modify_file.filename,
                    "pos": [add_method.start_line - 1, add_method.end_line],
                })

            modify_files.append({'commitId': commit_id,
                                 'filePath': modify_file.new_path,
                                 'fileContext': modify_file.source_code})

            commit_statics[commit_id] = output_path + "commit_chunk_" + str(j) + ".jsonl"

        if len(modify_files) > 200:
            file_path = output_path + "commit_chunk_" + str(j) + ".jsonl"
            with jsonlines.open(file_path, "w") as writer:
                writer.write_all(modify_files)
            modify_files = []
            j += 1

    if len(commit_statics) < 10:
        return None

    if modify_files:
        file_path = output_path + "commit_chunk_" + str(j) + ".jsonl"
        with jsonlines.open(file_path, "w") as writer:
            writer.write_all(modify_files)

    with open(output_path + "commit_statistic.jsonl", 'w') as f:
        json.dump(commit_statics, f)

    return result


def extract_funcs_from_file(root_path, file_path, call_parser):
    rel_file_path = os.path.relpath(file_path, root_path)

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            source_code = file.read()
    except Exception:
        return [], {'file_path': rel_file_path, 'source_code': "", 'import_lines': [], 'import_range': []}

    if source_code is None or source_code == "":
        return [], {'file_path': rel_file_path, 'source_code': "", 'import_lines': [], 'import_range': []}

    method_calls, import_stmt_positions = call_parser.extract(source_code)
    analyser = lizard.analyze_file.analyze_source_code(rel_file_path, source_code)

    func_list = []

    for method in analyser.function_list:
        current_calls = []

        for method_call in method_calls:
            call_tag, call_func_name, call_stmt, line_no, col_no = method_call
            if method.start_line - 1 <= line_no < method.end_line:
                current_calls.append({'called_func_name': call_func_name,
                                      'call_stmt': call_stmt,
                                      'type': call_tag,
                                      'line_no': line_no,
                                      'col_no': col_no})

        func_list.append({
            'file_path': rel_file_path,
            'name': method.name,
            'signature': method.long_name,
            'start_line': method.start_line-1,
            'end_line': method.end_line-1,
            'callee': current_calls,
        })

    import_lines = set()

    for node_pos in import_stmt_positions:
        start_point, end_point = node_pos
        import_lines.add(start_point[0])
        import_lines.add(end_point[0])

    import_lines = list(import_lines)

    file_info = {'file_path': rel_file_path,
                 'source_code': source_code.split('\n'),
                 'import_lines': import_lines,
                 'import_range': import_stmt_positions}

    return func_list, file_info


def extract_funcs_from_repo(repo_dir, call_parser):
    file_extension = call_parser.extension

    repo_func_list = []
    file_info_list = []

    walk = os.walk(repo_dir)
    for subdir, _, files in walk:
        for filename in files:
            path = os.path.join(subdir, filename)
            if filename.endswith(file_extension) and not filename.startswith("."):
                extract_result = extract_funcs_from_file(repo_dir, path, call_parser)
                if extract_result is None:
                    continue
                file_func_list, file_info = extract_result
                repo_func_list.extend(file_func_list)
                file_info_list.append(file_info)

    return repo_func_list, file_info_list


def analyze_ref_def_in_repo(repo_graph,
                            func_def_df,
                            func_ref_df,
                            skip_file_paths=[],
                            is_train=True):
    """
    :param repo_graph: The stack graph for this repo
    :param func_def_df: The dataframe which stores all function definitions.
    :param func_ref_df: The dataframe which stores all function calls to analyze.
    :param skip_file_paths: List[str] file paths needs to skip.
    :param is_train: if not is_train, do not consider the def after the ref.
    :return:
    """
    # definition list
    def_list = []
    # callee list
    callee_list = []
    def_sig_index_map = {}
    # (callee -> called)
    callee_edges = dict()
    # (called -> callee)
    called_edges = dict()

    call_info_list = []

    for row in func_ref_df.itertuples():
        func_calls = getattr(row, 'callee')
        file_path = getattr(row, 'file_path')
        func_sig = getattr(row, 'signature')

        if file_path in skip_file_paths:
            continue

        for func_call in func_calls:
            func_call['callee_func_index'] = int(row.Index)
            func_call['file_path'] = file_path
            func_call['signature'] = file_path + ":" + str(func_call['line_no']) + ":" + str(func_call['col_no'])
            func_call['parent_scope'] = func_sig
            call_info_list.append(func_call)

    total_call_len = len(call_info_list)
    call_info_df = pd.DataFrame.from_records(call_info_list)

    processed_call_num = 0
    bar = tqdm(total=total_call_len)

    while processed_call_num < total_call_len:
        call_df_chunk = call_info_df.iloc[processed_call_num: processed_call_num + 200]

        call_file_list = call_df_chunk['file_path'].tolist()
        line_no_list = call_df_chunk['line_no'].tolist()
        col_no_list = call_df_chunk['col_no'].tolist()

        ref_definitions = repo_graph.get_definitions(call_file_list,
                                                     line_no_list,
                                                     col_no_list)

        for index, definitions in enumerate(ref_definitions):
            if len(definitions) == 0:
                continue

            current_call_info = call_info_list[processed_call_num + index]

            possible_called_indexes = []
            for definition in definitions:
                def_sig, def_stmt = definition
                if def_stmt is None:
                    continue
                if def_sig in def_sig_index_map:
                    called_index = def_sig_index_map[def_sig]
                    called_def_info = def_list[called_index]
                    if called_def_info['file_path'] != call_file_list[index]:
                        ref_cross_file_def = True
                    else:
                        ref_cross_file_def = False

                else:
                    def_info = def_sig.split(":")
                    if def_info is None or len(def_info) != 4:
                        continue
                    def_path = def_info[0]
                    def_line_no, def_col_start, def_col_end = [int(s) for s in def_info[1:]]

                    define_type = "unk"
                    define_end_line = -1

                    if def_path != call_file_list[index]:
                        ref_cross_file_def = True
                    else:
                        ref_cross_file_def = False

                    if 'class' in def_stmt:
                        define_type = "class_declaration"
                    elif 'import' in def_stmt:
                        define_type = "import_declaration"
                    else:
                        if ref_cross_file_def:
                            define_end_line = get_define_body_end_line(func_def_df, def_path, def_line_no)
                        # The definition is in current file context, and we only definition before call.
                        elif def_line_no < line_no_list[index] or is_train:
                            define_end_line = get_define_body_end_line(func_ref_df, def_path, def_line_no)

                        if define_end_line == -1:
                            continue
                        define_type = "method_declaration"

                    def_info = {
                        'signature': def_sig,
                        'file_path': def_path,
                        'def_stmt': def_stmt,
                        'def_type': define_type,
                        'range': [def_line_no,
                                  define_end_line] if define_end_line != -1 else [def_line_no, def_line_no+1],
                    }

                    def_list.append(def_info)
                    called_index = len(def_list) - 1
                    def_sig_index_map[def_sig] = called_index

                possible_called_indexes.append((called_index, ref_cross_file_def))

            if len(possible_called_indexes) == 0:
                continue

            callee_list.append(current_call_info)
            callee_index = len(callee_list) - 1

            if callee_index not in callee_edges:
                callee_edges[callee_index] = []

            for called_index, ref_cross_file_def in possible_called_indexes:
                if called_index not in called_edges:
                    called_edges[called_index] = []

                callee_edges[callee_index].append((called_index, ref_cross_file_def))
                called_edges[called_index].append((callee_index, ref_cross_file_def))

        processed_call_num += len(line_no_list)
        bar.update(len(line_no_list))
        bar.set_description(f" process {processed_call_num}/{total_call_len} calls.")

    result = {
        'def_list': def_list,
        'callee_list': callee_list,
        'callee_edges': callee_edges,
        'called_edges': called_edges,
        'def_sig_index_map': def_sig_index_map,
    }

    return result


# def get_imported_definition(search_df, import_stmt, project_path, extension, call_name):
#     # 分几种情况
#     # 1. from 目录 import 方法, 在该目录下所有文件中查找(方法名)
#     # 2. from 文件 import 方法, 在该文件下所有方法中查找
#     # 3. import 方法, 回退到上一目录，查找是否存在该文件, 这个就先不考虑了
#     # 4. 查找文件
#     if 'from' in import_stmt:
#         import_stmt = import_stmt.replace('from', '')
#         import_index = import_stmt.index('import')
#         import_stmt = import_stmt[0: import_index].strip()
#     else:
#         import_stmt = import_stmt.replace('import', '')
#
#     import_path = project_path + import_stmt.replace(".", os.sep)
#     final_path = None
#
#     if os.path.exists(import_path):
#         final_path = import_path
#
#     if os.path.exists(import_path + "." + extension):
#         final_path = import_path + "." + extension
#
#     if final_path is None:
#         return None
#
#     condition = (search_df['file_path'].str.startswith(final_path) & (search_df['name'] == call_name))
#     method_def = search_df.loc[condition].iloc[0] if condition.any() else None
#     return method_def


def get_define_body_end_line(search_df, def_path, def_line_no):
    condition = (search_df['file_path'] == def_path) & (search_df['start_line'] == def_line_no)
    end_line = search_df.loc[condition, 'end_line'].iloc[0] if condition.any() else -1
    end_line = int(end_line)
    return end_line


def get_commits_by_path(path):
    results = {}
    with open(path, "r") as f:
        for line in f:
            line = json.loads(line)
            commit_id = line['commitId']
            filePath = line['filePath']
            content = line['fileContext']
            if commit_id not in results:
                results[commit_id] = {}
            if filePath not in results[commit_id]:
                results[commit_id][filePath] = content

    return results












