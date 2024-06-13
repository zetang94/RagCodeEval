import json
import os
import warnings
from math import ceil
from generation import parallel_generations
from ftp_utils import ftpconnect, uploadfile

_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
The "code_eval"/"apps_metric" you are about to use, execute untrusted 
model-generated code in Python.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network. For more
information on how OpenAI sandboxes its code, see the paper "Evaluating Large
Language Models Trained on Code" (https://arxiv.org/abs/2107.03374).
Once you have read this disclaimer and taken appropriate precautions, set the argument 
"allow_code_execution" to True.
################################################################################\
"""


class Evaluator:
    def __init__(self, accelerator, model, tokenizer, args):
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        # setup arguments
        self.metric_output_path = args.metric_output_path

    def generate_text(self, task):
        dataset = task.get_dataset()
        n_tasks = len(dataset)

        n_copies = ceil(self.args.n_samples / self.args.batch_size)
        tokenized_dataset = task.tokenize(n_copies, self.accelerator.state.num_processes)

        generations = parallel_generations(
            task,
            tokenized_dataset,
            self.accelerator,
            self.model,
            self.tokenizer,
            n_tasks=n_tasks,
            n_copies=n_copies,
            args=self.args,
        )
        references = [task.get_reference(dataset[i]) for i in range(n_tasks)]

        if len(generations[0]) > self.args.n_samples:
            generations = [l[: self.args.n_samples] for l in generations]
            warnings.warn(
                f"Number of tasks wasn't proportional to number of devices, "
                f"we removed extra predictions to only keep nsamples={self.args.n_samples}"
            )
        return generations, references, dataset

    def evaluate(self, task):
        generations, references, dataset = self.generate_text(task)

        if self.accelerator.is_main_process:
            if not self.args.load_generations_path:
                if self.args.save_generations:
                    with open(self.args.save_generations_path, "w") as fp:
                        json.dump(generations, fp)
                        print(
                            f"generations were saved at {self.args.save_generations_path}"
                        )
                    if self.args.use_codalab:
                        print("Beginning File Transfer.")
                        ftp = ftpconnect("47.113.193.20", "tangze", "soft12")
                        uploadfile(ftp, f"/generation/{self.args.output_file_name}",
                                   self.args.save_generations_path)
                        ftp.quit()
                        print("END Transfer.")
                        os.remove(self.args.save_generations_path)
                if self.args.save_references:
                    with open("references.json", "w") as fp:
                        json.dump(references, fp)
                        print("references were saved at references.json")

            # make sure tokenizer plays nice with multiprocessing
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            print("Evaluating generations...")
            results = task.process_results(generations, references, dataset)

            return results




