from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import pandas as pd
import requests
import torch
import tqdm
import numpy as np
import math


def cos_sim(a, b):
    """Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.

    Return:
        Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """  # noqa: D402
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def dot_score(a: torch.Tensor, b: torch.Tensor):
    """Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))


# From https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/custom_metrics.py#L4
def mrr(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: List[int],
) -> Tuple[Dict[str, float]]:
    MRR = {}

    for k in k_values:
        MRR[f"MRR@{k}"] = 0.0

    k_max, top_hits = max(k_values), {}
    logging.info("\n")

    for query_id, doc_scores in results.items():
        top_hits[query_id] = sorted(
            doc_scores.items(), key=lambda item: item[1], reverse=True
        )[0:k_max]

    for query_id in top_hits:
        query_relevant_docs = set(
            [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0]
        )
        for k in k_values:
            for rank, hit in enumerate(top_hits[query_id][0:k]):
                if hit[0] in query_relevant_docs:
                    MRR[f"MRR@{k}"] += 1.0 / (rank + 1)
                    break

    for k in k_values:
        MRR[f"MRR@{k}"] = round(MRR[f"MRR@{k}"] / len(qrels), 5)
        logging.info("MRR@{}: {:.4f}".format(k, MRR[f"MRR@{k}"]))

    return MRR


def recall_cap(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: List[int],
) -> Tuple[Dict[str, float]]:
    capped_recall = {}

    for k in k_values:
        capped_recall[f"R_cap@{k}"] = 0.0

    k_max = max(k_values)
    logging.info("\n")

    for query_id, doc_scores in results.items():
        top_hits = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[
            0:k_max
        ]
        query_relevant_docs = [
            doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0
        ]
        for k in k_values:
            retrieved_docs = [
                row[0] for row in top_hits[0:k] if qrels[query_id].get(row[0], 0) > 0
            ]
            denominator = min(len(query_relevant_docs), k)
            capped_recall[f"R_cap@{k}"] += len(retrieved_docs) / denominator

    for k in k_values:
        capped_recall[f"R_cap@{k}"] = round(capped_recall[f"R_cap@{k}"] / len(qrels), 5)
        logging.info("R_cap@{}: {:.4f}".format(k, capped_recall[f"R_cap@{k}"]))

    return capped_recall


def hole(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: List[int],
) -> Tuple[Dict[str, float]]:
    Hole = {}

    for k in k_values:
        Hole[f"Hole@{k}"] = 0.0

    annotated_corpus = set()
    for _, docs in qrels.items():
        for doc_id, score in docs.items():
            annotated_corpus.add(doc_id)

    k_max = max(k_values)
    logging.info("\n")

    for _, scores in results.items():
        top_hits = sorted(scores.items(), key=lambda item: item[1], reverse=True)[
            0:k_max
        ]
        for k in k_values:
            hole_docs = [
                row[0] for row in top_hits[0:k] if row[0] not in annotated_corpus
            ]
            Hole[f"Hole@{k}"] += len(hole_docs) / k

    for k in k_values:
        Hole[f"Hole@{k}"] = round(Hole[f"Hole@{k}"] / len(qrels), 5)
        logging.info("Hole@{}: {:.4f}".format(k, Hole[f"Hole@{k}"]))

    return Hole


def top_k_accuracy(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: List[int],
) -> Tuple[Dict[str, float]]:
    top_k_acc = {}

    for k in k_values:
        top_k_acc[f"Accuracy@{k}"] = 0.0

    k_max, top_hits = max(k_values), {}
    logging.info("\n")

    for query_id, doc_scores in results.items():
        top_hits[query_id] = [
            item[0]
            for item in sorted(
                doc_scores.items(), key=lambda item: item[1], reverse=True
            )[0:k_max]
        ]

    for query_id in top_hits:
        query_relevant_docs = set(
            [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0]
        )
        for k in k_values:
            for relevant_doc_id in query_relevant_docs:
                if relevant_doc_id in top_hits[query_id][0:k]:
                    top_k_acc[f"Accuracy@{k}"] += 1.0
                    break

    for k in k_values:
        top_k_acc[f"Accuracy@{k}"] = round(top_k_acc[f"Accuracy@{k}"] / len(qrels), 5)
        logging.info("Accuracy@{}: {:.4f}".format(k, top_k_acc[f"Accuracy@{k}"]))

    return top_k_acc


def get_rank_from_dict(
    dict_of_results: dict[str, float], doc_id: str
) -> Tuple[int, float]:
    tuple_of_id_score = dict_of_results.items()
    sorted_by_score = sorted(tuple_of_id_score, key=lambda x: x[1], reverse=True)
    for i, (id, score) in enumerate(sorted_by_score):
        if id == doc_id:
            return i + 1, score

    return len(sorted_by_score) + 1, 0


def evaluate_change(
    ori_run: dict[str, dict[str, float]],
    ins_run: dict[str, dict[str, float]],
    rev_run: dict[str, dict[str, float]],
    positive_qrels: dict[str, List[str]],
    negative_qrels: dict[str, List[str]],
) -> dict[str, float]:
    ins = []
    positive = []
    negative = []
    pos_neg = []

    
    # positive
    for qid in positive_qrels.keys():
        ori_qid_run = ori_run[qid]
        ins_qid_run = ins_run[qid]
        rev_qid_run = rev_run[qid]
        # summary positive
        pos_count = len(positive_qrels[qid])
        neg_count = len(negative_qrels[qid])
        sum_count = pos_count + neg_count
        for idx, relevant_doc in enumerate(positive_qrels[qid]):
            ori_rank, ori_score = get_rank_from_dict(ori_qid_run, relevant_doc)
            ins_rank, ins_score = get_rank_from_dict(ins_qid_run, relevant_doc)
            rev_rank, rev_score = get_rank_from_dict(rev_qid_run, relevant_doc)
            if ori_rank is None or ins_rank is None or rev_rank is None:
                ori_ins_rank_change = None
                ori_rev_rank_change = None
            else:
                ori_ins_rank_change = int(ori_rank - ins_rank) 
                ori_rev_rank_change = int(ori_rank - rev_rank) 
            positive.append(
                {
                    "qid": qid,
                    "doc_id": relevant_doc,
                    "ori_ins_rank_change": ori_ins_rank_change,
                    "ori_rev_rank_change":ori_rev_rank_change,
                    "relevance": 1,
                    "ori_rank": ori_rank,
                    "ins_rank": ins_rank,
                    "rev_rank": rev_rank,
                    "ori_score": ori_score,
                    "ins_score": ins_score,
                    "rev_score": rev_score,
                    "sum_count": sum_count
                }
            )


    #negative
    for qid in negative_qrels.keys():
        ori_qid_run = ori_run[qid]
        ins_qid_run = ins_run[qid]
        rev_qid_run = rev_run[qid]
        for idx, hard_doc in enumerate(negative_qrels[qid]):
            ori_rank, ori_score = get_rank_from_dict(ori_qid_run, hard_doc)
            ins_rank, ins_score = get_rank_from_dict(ins_qid_run, hard_doc)
            rev_rank, rev_score = get_rank_from_dict(rev_qid_run, hard_doc)
            if ori_rank is None or ins_rank is None or rev_rank is None:
                ori_ins_rank_change = None
                ori_rev_rank_change = None
            else:
                ori_ins_rank_change = int(ori_rank - ins_rank) 
                ori_rev_rank_change = int(ori_rank - rev_rank)
            negative.append(
                {
                    "qid": qid,
                    "doc_id": hard_doc,
                    "ori_ins_rank_change": ori_ins_rank_change,
                    "ori_rev_rank_change":ori_rev_rank_change,
                    "relevance": 0,
                    "ori_rank": ori_rank,
                    "ins_rank": ins_rank,
                    "rev_rank": rev_rank,
                    "ori_score": ori_score,
                    "ins_score": ins_score,
                    "rev_score": rev_score,
                }
            )
    
    #Calculate positive-negative
    for qid in positive_qrels.keys():
        ori_qid_run = ori_run[qid]
        ins_qid_run = ins_run[qid]
        # Get the rankings and scores of all positive documents and negative documents
        positive_ranks_scores = {pos_doc: get_rank_from_dict(ins_qid_run, pos_doc) for pos_doc in positive_qrels[qid]}
        negative_ranks_scores = {neg_doc: get_rank_from_dict(ins_qid_run, neg_doc) for neg_doc in negative_qrels[qid]}
        for pos_doc, (pos_rank, pos_score) in positive_ranks_scores.items():
            for neg_doc, (neg_rank, neg_score) in negative_ranks_scores.items():
                if pos_rank is None or neg_rank is None:
                    pos_neg_change = None
                else:
                    pos_neg_change = int(pos_rank - neg_rank)
                pos_neg.append({
                    "qid": qid,
                    "pos_id": pos_doc,
                    "pos_rank": pos_rank,
                    "neg_id": neg_doc,  
                    "neg_rank": neg_rank,
                    "pos_neg_score": pos_neg_change
                })
                

    # we now have a DF of [qid, doc_id, change] to run our calculations with
    pmrr_df = pd.DataFrame(negative)
    positive_df = pd.DataFrame(positive)
    negative_df = pd.DataFrame(negative)
    pos_neg_df = pd.DataFrame(pos_neg)
    # easy_neg_df = pd.DataFrame(easy_neg)
    
    # p-MRR
    pmrr_df["p-MRR"] = pmrr_df.apply(lambda x: pMRR_score(x), axis=1) 
    pmrr_qid_wise = pmrr_df.groupby("qid").agg({"p-MRR": "mean"})
    pmrr_qid_wise = pmrr_qid_wise.sort_index()

    # WISE
    positive_df["WISE"] = positive_df.apply(lambda x: WISE_score(x), axis=1) 
    WISE_qid_wise = positive_df.groupby("qid").agg({"WISE": "mean"})
    WISE_qid_wise = WISE_qid_wise.sort_index()
    positive_df["ideal-WISE"] = positive_df.apply(lambda x: ideal_WISE_score(x), axis=1) 
    ideal_WISE_qid_wise = positive_df.groupby("qid").agg({"ideal-WISE": "mean"})
    ideal_WISE_qid_wise = ideal_WISE_qid_wise.sort_index()

    # SICR
    sicr_results, sicr_score = SICR_score(positive)

    # positive ranking statistics
    positive_df["ori-pos-rank"] = positive_df.apply(lambda x: x["ori_rank"], axis=1) 
    ori_pos_rank = positive_df.groupby("qid").agg({"ori-pos-rank": "mean"})  
    positive_df["ins-pos-rank"] = positive_df.apply(lambda x: x["ins_rank"], axis=1) 
    ins_pos_rank = positive_df.groupby("qid").agg({"ins-pos-rank": "mean"})  
    positive_df["rev-pos-rank"] = positive_df.apply(lambda x: x["rev_rank"], axis=1) 
    rev_pos_rank = positive_df.groupby("qid").agg({"rev-pos-rank": "mean"})  

    # negative ranking statistics
    negative_df["ori-neg-rank"] = negative_df.apply(lambda x: x["ori_rank"], axis=1) 
    ori_neg_rank = negative_df.groupby("qid").agg({"ori-neg-rank": "mean"})  
    negative_df["ins-neg-rank"] = negative_df.apply(lambda x: x["ins_rank"], axis=1) 
    ins_neg_rank = negative_df.groupby("qid").agg({"ins-neg-rank": "mean"})  
    negative_df["rev-neg-rank"] = negative_df.apply(lambda x: x["rev_rank"], axis=1) 
    rev_neg_rank = negative_df.groupby("qid").agg({"rev-neg-rank": "mean"})


    # positive-negative ranking statistics
    pos_neg_df["pos-neg-rank"] = pos_neg_df.apply(lambda x: x["pos_neg_score"], axis=1) 
    pos_neg_rank = pos_neg_df.groupby("qid").agg({"pos-neg-rank": "mean"})  
    

    return {
        "p-MRR": pmrr_qid_wise["p-MRR"].mean(),
        # "p-MRR": pmrr_qid_wise["p-MRR"].to_dict(),
        "WISE": WISE_qid_wise["WISE"].mean(),
        "ideal_WISE": ideal_WISE_qid_wise["ideal-WISE"].mean(),
        "SICR": sicr_score,
        "ori_pos_rank":ori_pos_rank["ori-pos-rank"].mean(),
        "ins_pos_rank":ins_pos_rank["ins-pos-rank"].mean(),
        "rev_pos_rank":rev_pos_rank["rev-pos-rank"].mean(),
        "ori_neg_rank":ori_neg_rank["ori-neg-rank"].mean(),
        "ins_neg_rank":ins_neg_rank["ins-neg-rank"].mean(),
        "rev_neg_rank":rev_neg_rank["rev-neg-rank"].mean(),
        "pos_neg_rank":pos_neg_rank["pos-neg-rank"].mean(),
    }


def pMRR_score(x: dict[str, float]) -> float:
    if x["ori_rank"] >= x["ins_rank"]:
        return ((1 / x["ori_rank"]) / (1 / x["ins_rank"])) - 1
    else:
        return 1 - ((1 / x["ins_rank"]) / (1 / x["ori_rank"]))


def WISE_score(x: dict[str, float]) -> float:
    if x["ori_rank"] is None or x["ins_rank"] is None or x["rev_rank"] is None:
        return None
    if x["ins_rank"] <= x["ori_rank"] < x["rev_rank"]:
        if x["ori_rank"] < x["sum_count"] and x["ins_rank"] == 1:
            return 1
        elif (x["ori_rank"] <= 20):
            return (1 - math.sqrt(x["ori_rank"] - x["ins_rank"]) / 20) * (1 / math.sqrt(x["ins_rank"]))
        elif 20 < x["ori_rank"]:
            return 0.01 
    else:
        # if x["ins_rank"] < x["ori_rank"] < x["rev_rank"]:
        #     return -1
        if x["rev_rank"] < x["ori_rank"] < x["ins_rank"]:
            return -1
        elif x["ori_rank"] <= x["ins_rank"]:
            return ((x["ori_rank"] - x["ins_rank"]) / x["ins_rank"])
        elif x["rev_rank"] <= x["ori_rank"]: 
            return ((x["rev_rank"] - x["ori_rank"]) / x["ori_rank"])

def ideal_WISE_score(x: dict[str, float]) -> float:
    if x["ori_rank"] is None or x["ins_rank"] is None or x["rev_rank"] is None:
        return None
    x["ideal_ins_rank"] = 1
    x["ideal_rev_rank"] = x["ori_rank"] + 1
    if x["ideal_ins_rank"] <= x["ori_rank"] < x["ideal_rev_rank"]:
        if x["ori_rank"] < x["sum_count"] and x["ideal_ins_rank"] == 1:
            return 1
        elif (x["ori_rank"] <= 20):
            return (1 - math.sqrt(x["ori_rank"] - x["ideal_ins_rank"]) / 20) * (1 / math.sqrt(x["ideal_ins_rank"]))
        elif 20 < x["ori_rank"]:
            return 0.01 
        
from collections import defaultdict

def SICR_score(positive):
    results = {}
    total_increase = 0
    qid_grouped_pos = defaultdict(list)

    # Group positive by qid
    for pos in positive:
        qid_grouped_pos[pos["qid"]].append(pos)

    # Score each qid
    for qid in qid_grouped_pos.keys():
        gold_group = qid_grouped_pos[qid]

        for gold in gold_group:
            increase_score = 0
            increase = 0

            gold_ori_rank = gold["ori_rank"]
            gold_ins_rank = gold["ins_rank"]
            gold_rev_rank = gold["rev_rank"]
            gold_ori_score = gold["ori_score"]
            gold_ins_score = gold["ins_score"]
            gold_rev_score = gold["rev_score"]

            if gold_ori_rank is None or gold_ins_rank is None or gold_rev_rank is None or gold_ori_score is None or gold_ins_score is None or gold_rev_score is None:
                continue
            if gold_ori_rank > 1:
                if (gold_ori_rank > gold_ins_rank) and (gold_rev_rank > gold_ori_rank) and (gold_ins_score > gold_ori_score) and (gold_ori_score > gold_rev_score):
                    increase_score += 1

            if gold_ori_rank == 1:
                if (gold_ins_rank==1) and (gold_rev_rank > gold_ori_rank) and (gold_ins_score >= gold_ori_score) and (gold_ori_score > gold_rev_score):
                    increase_score += 1

            increase += increase_score

        # average
        num_gold = len(gold_group)
        avg_increase_score = increase / num_gold if num_gold > 0 else 0

        results[qid] = {
            "increase": avg_increase_score,
        }

        total_increase += avg_increase_score

    num_query = len(qid_grouped_pos)
    SICR = total_increase / num_query if num_query > 0 else 0

    return results, SICR


# https://stackoverflow.com/a/62113293
def download(url: str, fname: str):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm.tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
