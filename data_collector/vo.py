

class Sample:
    def __init__(self, prompt, ground_truth, right_context, delete_import_lines=[],
                 delete_line_nos=[], history_callees=[], def_meta=[], file_meta=None):
        self.prompt = prompt
        self.ground_truth = ground_truth
        self.right_context = right_context
        self.delete_import_lines = delete_import_lines
        self.delete_line_nos = delete_line_nos

        self.history_callees = history_callees
        self.def_meta = def_meta
        self.file_meta = file_meta

    def to_dict(self):
        return {
            'prompt': self.prompt,
            'ground_truth': self.ground_truth,
            'right_context': self.right_context,
            'delete_import_lines': self.delete_import_lines,
            'delete_line_nos': self.delete_line_nos,
            'history_callees': [callee.to_dict() for callee in self.history_callees],
            'definitions': [definition.to_dict() for definition in self.def_meta],
            'file_meta': None if self.file_meta is None else self.file_meta.to_dict()
        }

    @staticmethod
    def from_dict(d):
        history_callees = d['history_callees']
        called_def = d['called_def']
        file_meta = d['file_meta']

        history_callees = [CalleeMeta.from_dict(callee) for callee in history_callees]
        called_def = [DefMeta.from_dict(definition) for definition in called_def]
        file_meta = None if file_meta is None else FileMeta.from_dict(file_meta)

        return Sample(d['prompt'], d['ground_truth'], d['right_context'],
                      d['delete_import_lines'], d['delete_line_nos'],
                      history_callees, called_def, file_meta)


class CalleeMeta:
    def __init__(self, called_func_name, file_path, callee_stmt, callee_range):
        self.called_func_name = called_func_name
        self.file_path = file_path
        self.callee_stmt = callee_stmt
        self.callee_range = callee_range

    def to_dict(self):
        return {
            'called_func_name': self.called_func_name,
            'file_path': self.file_path,
            'stmt': self.callee_stmt,
            'range': self.callee_range
        }

    @staticmethod
    def from_dict(d):
        return CalleeMeta(d['called_func_name'], d['file_path'], d['stmt'], d['range'])


class DefMeta:
    def __init__(self, file_path, def_stmt, def_type, def_range):
        self.file_path = file_path
        self.def_stmt = def_stmt
        self.def_type = def_type
        self.def_range = def_range

    def to_dict(self):
        return {
            'file_path': self.file_path,
            'stmt': self.def_stmt,
            'type': self.def_type,
            'range': self.def_range
        }

    @staticmethod
    def from_dict(d):
        return DefMeta(d['file_path'], d['stmt'], d['type'], d['range'])


class FileMeta:
    def __init__(self, repo_name, file_path, commit_id, lifecycle):
        self.repo_name = repo_name
        self.file_path = file_path
        self.commit_id = commit_id
        self.lifecycle = lifecycle

    def to_dict(self):
        return {
            'repo_name': self.repo_name,
            'file_path': self.file_path,
            'commit_id': self.commit_id,
            'lifecycle': self.lifecycle
        }

    @staticmethod
    def from_dict(d):
        return FileMeta(d['repo_name'], d['file_path'], d['commit_id'], d['lifecycle'])
