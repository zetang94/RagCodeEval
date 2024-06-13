import json
import re
import subprocess
import time


class StackGraphHelper:
    def __init__(self, lang, path="/Users/tangze/study/rust_projects/stack-graphs/target/debug/"):
        langs = {
            "Java": path + "tree-sitter-stack-graphs-java",
            "Python": path + "tree-sitter-stack-graphs-python",
            "JavaScript": path + "tree-sitter-stack-graphs-javascript",
            "TypeScript": path + "tree-sitter-stack-graphs-typescript"
        }

        patterns = {
            "Python": re.compile(r"\b(import|def|class)\b"),
            "JavaScript": re.compile(r"\b(import|function|class)\b"),
            "TypeScript": re.compile(r"\b(import|function|class)\b"),
        }

        self.cmd_path = langs[lang]
        self.lang = lang
        self.pattern = patterns[lang] if lang != "Java" else None
        self.repo_path = None
        self.extension = None

    def set_current_repo_path(self, repo_path, extension):
        self.repo_path = repo_path
        self.extension = extension

    def clean(self):
        command = [self.cmd_path, 'clean', '--all']
        return self.run(command, timeout=60 * 30)

    def index_repo(self, repo_path):
        command = [self.cmd_path, "index", repo_path]
        result = self.run(command, timeout=60 * 30)

        if result is None:
            return None

        ignore_files = []
        file_ignore_str = '<FileIgnore>'
        str_len = len(file_ignore_str)

        if file_ignore_str in result:
            result_lines = result.split("\n")
            for line in result_lines:
                if line.startswith(file_ignore_str):
                    ignore_files.append(line[str_len:])

        return ignore_files

    @staticmethod
    def source_pos_to_str(source_pos, starts_with_zero=True):
        try:
            line_no = source_pos['line_no']
            start_col = source_pos['column_range']['start']

            if not starts_with_zero:
                line_no += 1
                start_col += 1

            sig = source_pos['path'] + ":" + str(line_no) + ":" + str(start_col)
            return sig
        except Exception as e:
            #print(e)
            return ""

    def get_definitions(self, file_paths, line_nos, cols):
        ref_signatures = []
        for file_path, line_no, col_no in zip(file_paths, line_nos, cols):
            ref_signature = file_path + ":" + str(line_no + 1) + ":" + str(col_no + 1)
            ref_signatures.append(ref_signature)

        ref_definitions = [set() for _ in ref_signatures]

        command = [self.cmd_path, 'query', 'definition', self.repo_path]
        command.extend(ref_signatures)

        result = self.run(command)

        if result is None:
            return ref_definitions

        if '<query-ok>' in result:
            result_lines = result.split("\n")
            total_len = len(result_lines)
            i = 0
            while i < total_len:
                if '\"reference\"' in result_lines[i]:
                    try:
                        result_line = json.loads(result_lines[i])
                        if 'definitions' in result_line and 'reference' in result_line:
                            ref_signature = self.source_pos_to_str(result_line['reference'], False)
                            ref_index = ref_signatures.index(ref_signature)
                            if ref_index > -1:
                                for d in result_line['definitions']:
                                    def_stmt = d['def_stmt']

                                    # For language like python, the definition might be the variable
                                    # So needs to ignore this situation.
                                    if self.lang != 'Java' and def_stmt is not None:
                                        match = self.pattern.findall(def_stmt)
                                        if not match:
                                            continue

                                    ref_definitions[ref_index].add(
                                        (self.source_pos_to_str(d) + ":" + str(d['column_range']['end']),
                                         def_stmt)
                                    )

                    except Exception as e:
                        print(e)
                        i += 1
                        continue
                i += 1

        return ref_definitions

    @staticmethod
    def run(commands, timeout=240):
        process = subprocess.Popen(commands, start_new_session=True,
                                   stdout=subprocess.PIPE)
        try:
            stdout, _ = process.communicate(timeout=timeout)
            output = stdout.decode("utf-8")
            return output
        except subprocess.TimeoutExpired:
            process.kill()
            print(f'Timeout for {commands} ({timeout}s) expired')
            return None
        except Exception as e:
            process.kill()
            print(e)
            return None


if __name__ == '__main__':
    helper = StackGraphHelper("Python")
    helper.clean()
    results = helper.index_repo("/Users/tangze/study/py_projects/project-method-miner/test_stack_graph_py")
    print(results)
    file_path = ["/Users/tangze/study/py_projects/project-method-miner/test_stack_graph_py/chef.py"]
    line_no = [8]
    col_no = [4]

    print(helper.get_definitions(file_path, line_no, col_no))
    # helper = StackGraphHelper("Java")
    # helper.clean()
    # helper.index_repo("/Users/tangze/study/py_projects/project-method-miner/test_stack_graph_java")
    # file_path = ["/Users/tangze/study/py_projects/project-method-miner/test_stack_graph_java/kitchen.java"]
    # line_no = [6]
    # col_no = [8]
    # print(helper.get_definitions(file_path, line_no, col_no))
