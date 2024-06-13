import json
import os

import torch
import transformers
from accelerate import Accelerator
from huggingface_hub import snapshot_download
from transformers import HfArgumentParser, AutoModelForCausalLM, AutoTokenizer
from arguments import EvalArguments
from code_completion import NextLineCompletionTask
from evaluator import Evaluator
from ftp_utils import ftpconnect, uploadfile



def get_hf_model(repo_name, cache="/caches", use_auth_token=None, revision=None):
    print(f'load {repo_name} from cache.')
    if revision is None:
        model_path = snapshot_download(repo_name, cache_dir=cache,
                                       local_files_only=True,
                                       use_auth_token=use_auth_token)
    else:
        model_path = snapshot_download(repo_name, cache_dir=cache,
                                       local_files_only=True,
                                       use_auth_token=use_auth_token, revision=revision)

    print(model_path)
    return model_path


def parse_args():
    parser = HfArgumentParser(EvalArguments)

    parser.add_argument(
        "--model",
        default="codeparrot/codeparrot-small",
        help="Model to evaluate, provide a repo name in Hugging Face hub or a local path",
    )

    parser.add_argument(
        "--test_file",
        default=None,
        help="Evaluation performance on test file.",
    )

    parser.add_argument(
        "--language",
        default=None,
        help="The code language to evaluate."
    )

    parser.add_argument(
        "--max_sequence_len",
        type=int,
        default=2048,
        help="Maximum length of generated sequence (prompt+generation)",
    )

    parser.add_argument(
        "--max_gen_len",
        type=int,
        default=50,
        help="Maximum length of generating."
    )

    parser.add_argument(
        "--k_shots",
        type=int,
        default=5,
        help="The number of shot examples."
    )

    parser.add_argument(
        "--max_cfg_len",
        type=int,
        default=512,
        help="The max length of concat cross-file code contexts."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation on each worker, can be larger for HumanEval",
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        help="Model precision, from: fp32, fp16 or bf16",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to solve and evaluate from the benchmark",
    )
    parser.add_argument(
        "--use_codalab",
        action="store_true",
        help="If use codalab, save outputs into ftp server(If the output files are too large, "
             "the codalab cannot show results correctly(personal view).)"
    )

    parser.add_argument(
        "--postprocess",
        action="store_false",
        help="Postprocess model outputs before execution, always on except during generation tests",
    )
    parser.add_argument(
        "--generation_only",
        action="store_true",
        help="Do code generation but no evaluation",
    )
    parser.add_argument(
        "--load_generations_path",
        type=str,
        default=None,
        help="Path of file with previously generated solutions, if provided generation is skipped and only evaluation is done",
    )
    parser.add_argument(
        "--metric_output_path",
        type=str,
        default="evaluation_results.json",
        help="Path to save the results",
    )
    parser.add_argument(
        "--save_generations",
        action="store_true",
        help="Whether to save code generations",
    )
    parser.add_argument(
        "--save_generations_path",
        type=str,
        default="generations.json",
        help="Path for saving the code generations",
    )

    parser.add_argument(
        "--save_references",
        action="store_true",
        help="Whether to save code generations",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model.split('/')[1]
    k_shots = str(args.k_shots)
    test_file_name, _ = os.path.splitext(os.path.basename(args.test_file))
    args.output_file_name = f"{model_name}_{k_shots}_shots_{test_file_name}_generation.json"

    transformers.logging.set_verbosity_error()

    accelerator = Accelerator()
    if accelerator.is_main_process:
        print(f"Evaluate {args.language} on file {args.test_file}.")

    results = {}
    if args.load_generations_path:
        if accelerator.is_main_process:
            print("evaluation only mode")
        evaluator = Evaluator(accelerator, None, None, args)

        n_tasks = args.limit if args.limit else None
        task = NextLineCompletionTask(args, None, n_tasks)
        results = evaluator.evaluate(task)
    else:
        # here we generate code and save it (evaluation is optional but True by default)
        dict_precisions = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        if args.precision not in dict_precisions:
            raise ValueError(
                f"Non valid precision {args.precision}, choose from: fp16, fp32, bf16"
            )
        print(f"Loading tokenizer and model (in {args.precision})")

        auth_token = None
        if args.model.startswith('bigcode'):
            auth_token = "your_auth_token"

        if args.model == 'facebook/incoder-6B':
            model = AutoModelForCausalLM.from_pretrained(
                get_hf_model(args.model, revision="float16"),
                torch_dtype=torch.float16,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                get_hf_model(args.model, revision="float16"),
                truncation_side="left",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                get_hf_model(args.model, use_auth_token=auth_token),
                torch_dtype=dict_precisions[args.precision],
            )
            tokenizer = AutoTokenizer.from_pretrained(
                get_hf_model(args.model, use_auth_token=auth_token),
                truncation_side="left",
            )

        if not tokenizer.eos_token:
            if tokenizer.bos_token:
                tokenizer.eos_token = tokenizer.bos_token
                print("bos_token used as eos_token")
            else:
                raise ValueError("No eos_token or bos_token found")
        tokenizer.pad_token = tokenizer.eos_token
        evaluator = Evaluator(accelerator, model, tokenizer, args)

        n_tasks = args.limit if args.limit else None
        task = NextLineCompletionTask(args, tokenizer, n_tasks)

        if args.generation_only:
            if accelerator.is_main_process:
                print("generation mode only")
            generations, references = evaluator.generate_text(task)
            if accelerator.is_main_process:
                with open(args.save_generations_path, "w") as fp:
                    json.dump(generations, fp)

                if args.use_codalab:
                    # Use ftp server to save output files.
                    print("Beginning File Transfer.")
                    ftp = ftpconnect("0.0.0.0", "user", "passwd")
                    uploadfile(ftp, f"/generation/{args.output_file_name}",
                               args.save_generations_path)
                    ftp.quit()
                    print("END Transfer.")
                    os.remove(args.save_generations_path)

                if args.save_references:
                    with open("references.json", "w") as fp:
                        json.dump(references, fp)
                        print("references were saved")
        else:
            results = evaluator.evaluate(task)

    if accelerator.is_main_process:
        results["config"] = {
            "model": args.model,
            "testfile": args.test_file,
            "shot": args.k_shots,
            "temperature": args.temperature,
            "n_samples": args.n_samples,
        }
        if not args.generation_only:
            dumped = json.dumps(results, indent=2)
            print(results["config"])
            print(results["mean"])

            with open(args.metric_output_path, "w") as f:
                f.write(dumped)

            with open('score.json', 'w') as f:
                score = {
                    'mean': results['mean'],
                    'config': results['config']
                }
                json.dump(score, f)


if __name__ == "__main__":
    main()

