"""Standard ranking metrics, implemented directly so every formula is visible in the report.

nDCG@k  : rewards relevant documents near the top (log2 discount), normalised by the ideal ranking.
Recall@k: share of a query's relevant documents found in the top k.
P@k     : share of the top k that is relevant (low by construction when a query has ~1 relevant doc).
MRR@k   : 1 / rank of the first relevant document (0 if none in the top k).
All are averaged over queries.
"""

import math
import time

import numpy as np


def _dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def query_metrics(ranked_doc_ids, relevant, ks=(1, 3, 5, 10, 100)):
    """ranked_doc_ids: list of doc ids, best first. relevant: {doc_id: grade}."""
    rel = {d: g for d, g in relevant.items() if g > 0}
    out = {}
    for k in ks:
        top = ranked_doc_ids[:k]
        hits = [rel.get(d, 0) for d in top]
        ideal = sorted(rel.values(), reverse=True)[:k]
        out[f"nDCG@{k}"] = _dcg(hits) / _dcg(ideal) if ideal else 0.0
        out[f"Recall@{k}"] = sum(1 for d in top if d in rel) / len(rel) if rel else 0.0
        out[f"P@{k}"] = sum(1 for d in top if d in rel) / k
    first = next((i for i, d in enumerate(ranked_doc_ids[:10]) if d in rel), None)
    out["MRR@10"] = 1.0 / (first + 1) if first is not None else 0.0
    return out


def run_retriever(retriever, corpus, k=100):
    """Search every labelled query. Returns ({qid: [doc_id, ...]}, mean latency in ms)."""
    run, t0 = {}, time.time()
    for qid, text in corpus.queries.items():
        run[qid] = [corpus.doc_ids[i] for i, _ in retriever.search(text, k)]
    return run, 1000 * (time.time() - t0) / max(len(corpus.queries), 1)


def per_query_scores(run, qrels, ks=(1, 3, 5, 10, 100)):
    return {qid: query_metrics(run.get(qid, []), qrels[qid], ks) for qid in qrels if qid in run}


def evaluate_run(run, qrels, metrics=("nDCG@10", "Recall@10", "Recall@100", "MRR@10", "P@10")):
    pq = per_query_scores(run, qrels)
    return {m: float(np.mean([s[m] for s in pq.values()])) for m in metrics}
