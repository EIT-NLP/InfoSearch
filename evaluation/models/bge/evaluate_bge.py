import os
import argparse

from flag_dres_model import FlagDRESModel
from mteb import MTEB

query_instruction_for_retrieval_dict = {
    "BAAI/bge-large-en": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-small-en": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-en-icl":"Represent this sentence for searching relevant passages: ",
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default="BAAI/bge-en-icl", type=str)
    parser.add_argument('--task_type', default=None, type=str, help="task type. Default is None, which means using all task types")
    parser.add_argument('--add_instruction', action='store_true', help="whether to add instruction for query")
    parser.add_argument('--pooling_method', default='cls', type=str)
    parser.add_argument('--output_dir', default='bge-icl-0.5', type=str, help="output folder for the results")
    parser.add_argument('--task_names', default=None, type=str, nargs='+', help="task names")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    model = FlagDRESModel(model_name_or_path=args.model_name_or_path,
                          normalize_embeddings=False,  # normlize embedding will harm the performance of classification task
                          query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                          pooling_method=args.pooling_method)

    if args.task_names is None:
        task_names = [t.metadata_dict["name"] for t in MTEB(task_types=['InstructionRetrieval']).tasks]
    else:
        task_names = args.task_names

    for task in task_names:
        if task in ['MSMARCOv2']:
            print('Skip task: {}, since it has no test split'.format(task))
            continue

        if 'CQADupstack' in task or task in ['Touche2020', 'SciFact', 'TRECCOVID', 'NQ',
                                             'NFCorpus', 'MSMARCO', 'HotpotQA', 'FiQA2018',
                                             'FEVER', 'DBPedia', 'ClimateFEVER', 'SCIDOCS', ]:
            if args.model_name_or_path not in query_instruction_for_retrieval_dict:
                if args.add_instruction:
                    instruction = "Represent this sentence for searching relevant passages: "
                else:
                    instruction = None
                print(f"{args.model_name_or_path} not in query_instruction_for_retrieval_dict, set instruction={instruction}")
            else:
                instruction = query_instruction_for_retrieval_dict[args.model_name_or_path]
        else:
            instruction = None

        model.query_instruction_for_retrieval = instruction

        evaluation = MTEB(tasks=[task], task_langs=['en'], eval_splits = ["test" if task not in ['MSMARCO'] else 'dev'], do_length_ablation=False)
        evaluation.run(model,
                       output_folder=args.output_dir,
                       save_corpus_embeddings=True)


