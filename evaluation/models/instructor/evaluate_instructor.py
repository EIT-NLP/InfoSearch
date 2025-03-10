import os
import sys
import logging
import argparse
from mteb import MTEB
from instructor_embedding import Instructor
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="hkunlp/instructor-xl",type=str)
    parser.add_argument('--output_dir', default="instructor-xl-0.5",type=str)
    parser.add_argument('--task_names', default=None,type=str, nargs='+')
    parser.add_argument('--cache_dir', default=None,type=str)
    parser.add_argument('--prompt', default=None,type=str)
    parser.add_argument('--split', default='test',type=str)
    parser.add_argument('--batch_size', default=16,type=int)
    args = parser.parse_args()

    # from tqdm import tqdm
    # from functools import partialmethod
    #
    # tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    model = Instructor(args.model_name, cache_folder=args.cache_dir)

    if args.task_names is None:
        task_names = [t.metadata_dict["name"] for t in MTEB(task_types=['InstructionRetrieval']).tasks]
    else:
        task_names = args.task_names

    for task in task_names:
        evaluation = MTEB(tasks=[task], task_langs=["en"], do_length_ablation=False)
        evaluation.run(model, output_folder=args.output_dir, eval_splits=[args.split],args=args, save_corpus_embeddings=True)

    print("--DONE--")