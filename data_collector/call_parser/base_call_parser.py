from abc import ABC, abstractmethod
from collections import Counter


class CallParser():
    __metaclass__ = ABC
    """A string holding the name of the language, ex: 'python' """

    @property
    @abstractmethod
    def language(self):
        pass

    """A string holding the file extension for the language, ex: '.java' """
    @property
    @abstractmethod
    def extension(self):
        pass

    @property
    @abstractmethod
    def language_library(self):
        pass

    @property
    @abstractmethod
    def parser(self):
        pass

    """The query that finds all the function calls and import statements in the file"""
    @property
    @abstractmethod
    def query(self):
        pass

    def extract(self, source_code):
        try:
            root_node = self.parser.parse(bytes(source_code, "utf8")).root_node
            lines = source_code.split("\n")
            captures = self.query.captures(root_node)

            results = []
            import_stmt_positions = []

            # ignore multi calls in the same line.
            counts = Counter([node.start_point[0] for node, _ in captures])
            duplicates = [item for item, count in counts.items() if count > 1]

            for node, tag in captures:
                line_no, col_no = node.start_point
                if line_no in duplicates:
                    continue

                call_func_name = pos_to_string(lines, (node.start_point, node.end_point))
                call_stmt = pos_to_string(lines, ((node.start_point[0], 0), (node.start_point[0], -1)))

                if tag == 'import':
                    import_stmt_positions.append((node.start_point, node.end_point))
                else:
                    results.append((tag, call_func_name, call_stmt, line_no, col_no))

            return results, import_stmt_positions
        except Exception as e:
            print(e)
            return [], []


def pos_to_string(lines, pos) -> str:
    start_point, end_point = pos

    if start_point[0] == end_point[0]:
        return lines[start_point[0]][start_point[1]:end_point[1]]
    ret = lines[start_point[0]][start_point[1]:] + "\n"
    ret += "\n".join([line for line in lines[start_point[0] + 1:end_point[0]]])
    ret += "\n" + lines[end_point[0]][:end_point[1]]
    return ret
