import os
import logging
import argparse
from huggingface_hub import login
from mteb import MTEB

logging.basicConfig(level=logging.INFO)
from reranker_models import *


logger = logging.getLogger("main")

# "meta-llama/Llama-2-7b-chat-hf", meta-llama/Meta-Llama-3.1-8B-Instruct,jhu-clsp/FollowIR-7B, mistralai/Mistral-7B-Instruct-v0.2,castorini/rank_zephyr_7b_v1_full

if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="mistralai/Mistral-7B-Instruct-v0.2", type=str)
    parser.add_argument("--output_dir", default="Mistral-7B-0.5", type=str)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--fp_options", default="bfloat16", type=str)
    parser.add_argument("--task_names", default=None, type=str, nargs='+')
    args = parser.parse_args()
    print(args)

    if args.model_name_or_path in MODEL_DICT:
        if args.model_name_or_path == "castorini/rank_zephyr_7b_v1_full":
            model = MODEL_DICT[args.model_name_or_path](args.model_name_or_path)
        else:
            model = MODEL_DICT[args.model_name_or_path](args.model_name_or_path, fp_options=args.fp_options)
    else:
        model = MODEL_DICT["custom_mistral"](args.model_name_or_path, fp_options=args.fp_options)

    if args.task_names is None:
        task_names = [t.metadata_dict["name"] for t in MTEB(task_types=['InstructionRetrieval']).tasks]
    else:
        task_names = args.task_names

    for task in task_names:
        logger.info(f"Running task: {task}")
        eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
        evaluation = MTEB(tasks=[task], task_langs=["en"], do_length_ablation=False)  # Remove "en" for running all languages
        task_name_for_scores = task.split("InstructionRetrieval")[0].lower()
        evaluation.run(model,
                       output_folder=args.output_dir,
                       eval_splits=eval_splits,
                       batch_size=args.batch_size,
                       top_k=100,
                       previous_results=f"https://hf-mirror.com/datasets/jianqunZ/{task_name_for_scores}/raw/main/empty_scores.json",
                        do_length_ablation=False)
