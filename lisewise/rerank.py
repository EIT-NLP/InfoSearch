from pathlib import Path
from dacite import from_dict
from rank_llm.data import DataWriter, Request
import json
import torch
from rank_llm.rerank import Reranker, get_openai_api_key
from rank_llm.rerank.listwise import (
    SafeOpenai,
    VicunaReranker,
    ZephyrReranker,
)


def main(args):
    model_path = args.model_path
    dataset = args.dataset

    context_size = args.context_size
    top_k_candidates = args.top_k_candidates
    device = "cuda" if torch.cuda.is_available() else "cpu"

    reranker = None
    model_name = ""

    if "zephyr" in model_path.lower():
        reranker = ZephyrReranker(model_path=model_path, context_size=context_size, device=device)
        model_name = "zephyr"
    elif "vicuna" in model_path.lower():
        reranker = VicunaReranker(model_path=model_path, context_size=context_size, device=device)
        model_name = "vicuna"
    elif "gpt" in model_path.lower():
        reranker = SafeOpenai(model=model_path, context_size=context_size, keys=get_openai_api_key())
        model_name = model_path

    if reranker is None:
        raise ValueError("Invalid model path provided. Please provide a valid model")

    # get retrieved results
    with open(dataset, "r") as read_file:
        data = json.load(read_file)

    retrieved_results = []
    for item in data:
        request = from_dict(data_class=Request, data=item)
        retrieved_results.append(request)

    # rerank the retrieved results
    rerank_results = reranker.rerank_batch(requests=retrieved_results, rank_end=top_k_candidates)

    # save the reranked results
    writer = DataWriter(rerank_results)
    Path(f"results/").mkdir(parents=True, exist_ok=True)

    # get current time
    import time
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    file_name = dataset.split("/")[-1].split(".")[0]

    writer.write_in_jsonl_format(f"results/{file_name}_{model_name}_{current_time}.jsonl")


def json_to_request(json_data: str) -> Request:
    return from_dict(data_class=Request, data=json_data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RankLLM Rerank")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model to use for reranking."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the dataset to rerank."
    )
    parser.add_argument(
        "--context_size",
        type=int,
        default=4096,
        help="The context size for the reranker."
    )
    parser.add_argument(
        "--top_k_candidates",
        type=int,
        default=100,
        help="The number of top candidates to retrieve."
    )

    args = parser.parse_args()
    main(args)
