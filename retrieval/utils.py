# Copyright Amazon.com, Inc. or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import re
import os
from nltk.tokenize import word_tokenize
import nltk

nltk.data.path.insert(0, "/root/nltk_data")


def tokenize_nltk(text):
    words = word_tokenize(text)
    output_list = []
    for w in words:
        w_list = re.findall(r'\w+', w)
        output_list.extend(w_list)
    return output_list


def file_distance(src_file, dest_file):
    distance = -1
    try:
        commonpath = os.path.commonpath([src_file, dest_file])
        rel_file1_path = os.path.relpath(src_file, commonpath)
        rel_file2_path = os.path.relpath(dest_file, commonpath)
        distance = rel_file1_path.count(os.sep) + rel_file2_path.count(os.sep)
    except Exception as e:
        # print(e, src_file, dest_file)
        pass

    return distance


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def find_files_within_distance_k(current_file_path, filelist, k):
    list_of_modules = []
    module_weight = []
    for filepath in filelist:
        if filepath != current_file_path:
            dist = file_distance(filepath, current_file_path)
            if dist == -1:
                continue
            elif dist <= k:
                list_of_modules.append(filepath)
                module_weight.append(dist)

    # sorting in ascending order
    list_of_modules = [x for _, x in sorted(zip(module_weight, list_of_modules))]
    return list_of_modules
