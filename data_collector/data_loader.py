import json

import jsonlines
import pandas as pd
import seaborn as sns
import functools
import numpy as np
import matplotlib.pyplot as plt


class RepoLoader:
    def __init__(self, data_path, meta_file_path):
        try:
            self.data = pd.read_json(meta_file_path, orient="records", lines=True)
        except Exception as e:
            print(f"Error to load the dataset meta file from path {meta_file_path}. ", e)

        self.data_path = data_path

    # Get selected repo dataset.
    def filter(self, repo_name=None, lang=None, lifecycle=None, filter_repos=[]):
        conditions = []
        if repo_name is not None:
            conditions.append(self.data["name"] == repo_name)
        if lang is not None:
            conditions.append(self.data['lang'] == lang)
        if lifecycle is not None:
            conditions.append(self.data['lifecycle'] == lifecycle)
        if len(filter_repos) != 0:
            conditions.append(~self.data['name'].isin(filter_repos))

        if len(conditions) == 0:
            filtered_data = self.data
        else:
            filtered_data = self.data[self.disjunction(*conditions)]

        return filtered_data

    def load_samples(self, repo_data_path, need_snap_files):
        try:
            repo_data = RepoData(self.data_path + repo_data_path, need_snap_files=need_snap_files)
            return repo_data
        except Exception as e:
            #print(f"Load error! {repo_data_path}, exception msg: {e}. ")
            return None

    @staticmethod
    def disjunction(*conditions):
        return functools.reduce(np.logical_and, conditions,)

    # Get the statistical of the snapshot.
    def snap_statistic(self, df, skipped_repo=set()):
        avg = [[], []]
        for index, row in df.iterrows():
            repo_path = row['dataset_path']
            repo_data = self.load_samples(repo_path, True)
            repo_name = row['name']

            has_ref = 0
            no_ref = 0

            if repo_data is None:
                skipped_repo.add(repo_name)
                continue

            called_def_signatures = []
            for _, row2 in repo_data.snap_definitions.iterrows():
                def_type = getattr(row2, "def_type")
                if def_type != "method_declaration":
                    continue
                file_path = getattr(row2, "file_path")
                def_range = json.loads(getattr(row2, "range"))
                called_def_signatures.append(file_path + ":" + str(def_range[0]) + ":" + str(def_range[1]))

            total = 0
            for _, row3 in repo_data.snap_functions.iterrows():
                file_path = getattr(row3, 'file_path')
                start_line = getattr(row3, 'start_line')
                end_line = getattr(row3, 'end_line')

                sig = file_path + ":" + str(start_line) + ":" + str(end_line)

                if sig not in called_def_signatures:
                    no_ref += 1
                else:
                    has_ref += 1

                total += 1

            has_ref = has_ref #/ total
            no_ref = no_ref #/ total

            avg[0].append(has_ref)
            avg[1].append(no_ref)

        avg = np.asarray(avg)

        return avg


    # Get the statistical of the test dataset.
    def statistic(self, df, skipped_repo=set()):
        avg = [[], [], [], []]

        for index, row in df.iterrows():
            repo_path = row['dataset_path']
            repo_data = self.load_samples(repo_path)
            repo_name = row['name']

            if repo_data is None:
                skipped_repo.add(repo_name)
                continue

            has_def_has_ref = 0
            has_def_no_ref = 0
            no_def_has_ref = 0
            no_def_no_ref = 0
            total = 0

            for sample in repo_data.test_data:
                if sample['has_callee']:
                    if sample['has_def']:
                        has_def_has_ref += 1
                        #total += 1
                    else:
                        no_def_has_ref += 1
                else:
                    if sample['has_def']:
                        has_def_no_ref += 1
                        #total += 1
                    else:
                        no_def_no_ref += 1

                total += 1

            if total == 0:
                skipped_repo.add(repo_name)
                continue

            has_def_has_ref = has_def_has_ref / total
            has_def_no_ref = has_def_no_ref / total
            no_def_has_ref = no_def_has_ref / total
            no_def_no_ref = no_def_no_ref / total

            avg[0].append(has_def_has_ref)
            avg[1].append(has_def_no_ref)
            avg[2].append(no_def_has_ref)
            avg[3].append(no_def_no_ref)

        avg = np.asarray(avg)

        return avg


class RepoData:
    def __init__(self, repo_data_path, need_snap_files=False):
        self.repo_data_path = repo_data_path

        if need_snap_files:
            self.snap_files = pd.read_csv(self.repo_data_path + "/Files.csv")
            self.snap_functions = pd.read_csv(self.repo_data_path + "/Functions.csv")
            self.snap_definitions = pd.read_csv(self.repo_data_path + "/Definitions.csv")

        self.test_data = []

        with jsonlines.open(self.repo_data_path + "/test.jsonl") as r:
            for line in r:
                self.test_data.append(line)
        self.test_meta_data = []
        with jsonlines.open(self.repo_data_path + "/test_meta.jsonl") as r:
            for line in r:
                self.test_meta_data.append(line)


def draw_plot(labels, language, *data):
    print(data)
    # 设置Seaborn的样式
    sns.set(style="whitegrid")

    # 绘制堆叠面积图
    plt.figure(figsize=(10, 6))
    plt.stackplot(*data, labels=labels, alpha=0.8)

    plt.legend(loc='upper left')
    plt.title(f"{language}")
    plt.xlabel("Lifecycle")
    plt.ylabel("Ratio")
    #plt.tight_layout()

    plt.savefig("output.png")
    plt.show()


if __name__ == "__main__":
    repo_loader = RepoLoader("../../", "data_builder/metadata/dataset_metadata.jsonl")

    lang = "Java"

    results = []

    lifecycles = ["Initiation", "Intermediate", "Closure"]
    skip_repo = set()

    for x in lifecycles:
        filtered_data = repo_loader.filter(lang=lang, lifecycle=x)
        repo_loader.snap_statistic(filtered_data, skip_repo)

    for x in lifecycles:
        filtered_data = repo_loader.filter(lang=lang, lifecycle=x, filter_repos=list(skip_repo))
        result = repo_loader.snap_statistic(filtered_data, )
        results.append(result)

    results = np.asarray(results)
    print("Total repo nums: ", results.shape[-1])
    results = np.mean(results, axis=-1)

    labels = ['Repo Call, history_callee=True', 'Repo Call, history_callee=False']
              #'third-party-call, history_callee=True', 'Third-party Call, history_callee=False']

    draw_plot(labels, lang, lifecycles, results[:, 0], results[:, 1],)
              #results[:, 2], results[:, 3])
