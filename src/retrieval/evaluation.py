"""Evaluation metrics for evidence retrieval: Recall@k, MRR@k, and nDCG@k."""

import math
from typing import Dict, List, Set, Tuple

from src.data.schema import FEVERRecord
from src.retrieval.retriever import BaseRetriever
from src.retrieval.schema import RetrievalMetrics, RetrievedPassage
from src.utils.logging import get_logger

logger = get_logger(__name__)


def evaluate_retrieval(
    retriever: BaseRetriever,
    records: List[FEVERRecord],
    top_k: int = 5
) -> RetrievalMetrics:
    """Evaluates retrieval quality against FEVER gold evidence.
    
    Relevance Definition:
    In FEVER, ground truth evidence is represented as disjunctions of conjunctions of
    (wikipedia_title, sentence_id) coordinates. A retrieved passage is considered a true positive (relevant)
    if its (title, sentence_id) coordinate matches any element in the record's gold evidence sets.
    
    Note: Evaluated strictly on verifiable claims (SUPPORTS and REFUTES) that possess non-null gold evidence.
    Claims with NOT ENOUGH INFO are tracked as unretrievable by definition.
    
    Metrics Computed:
    - Recall@k: Proportion of verifiable claims where at least one gold sentence is retrieved in top-k.
    - MRR@k: Mean Reciprocal Rank (1/rank of the first relevant passage in top-k).
    - nDCG@k: Normalized Discounted Cumulative Gain at truncation k.
    - Precision@k: Average fraction of retrieved passages in top-k that are gold.
    """
    verifiable_records = [r for r in records if r.has_gold_evidence()]
    total_queries = len(verifiable_records)

    if total_queries == 0:
        return RetrievalMetrics(
            total_queries=0,
            k=top_k,
            recall_at_k=0.0,
            mrr_at_k=0.0,
            ndcg_at_k=0.0,
            precision_at_k=0.0,
            unretrieved_queries=0
        )

    hits_count = 0
    reciprocal_ranks = []
    ndcg_scores = []
    precisions = []

    for rec in verifiable_records:
        gold_pairs: Set[Tuple[str, int]] = set()
        for g_set in rec.get_gold_evidence_sets():
            gold_pairs.update(g_set)

        if not gold_pairs:
            continue

        # Retriever sees ONLY the claim text
        retrieved: List[RetrievedPassage] = retriever.retrieve(rec.claim, top_k=top_k)

        # Check relevance of each retrieved passage
        relevance_binary = []
        first_hit_rank = None

        for idx, passage in enumerate(retrieved, start=1):
            coord = (passage.title, passage.sentence_id)
            is_rel = 1 if coord in gold_pairs else 0
            relevance_binary.append(is_rel)
            passage.is_gold = bool(is_rel)

            if is_rel and first_hit_rank is None:
                first_hit_rank = idx

        # Recall@k
        has_hit = 1 if any(relevance_binary) else 0
        hits_count += has_hit

        # MRR@k
        if first_hit_rank is not None:
            reciprocal_ranks.append(1.0 / first_hit_rank)
        else:
            reciprocal_ranks.append(0.0)

        # Precision@k
        prec = sum(relevance_binary) / float(top_k) if top_k > 0 else 0.0
        precisions.append(prec)

        # nDCG@k
        # DCG@k = sum(rel_i / log2(i + 1))
        dcg = sum(rel / math.log2(idx + 1) for idx, rel in enumerate(relevance_binary, start=1))

        # Ideal DCG (IDCG): all available gold items sorted at top ranks
        ideal_relevance = sorted([1] * min(len(gold_pairs), top_k) + [0] * max(0, top_k - len(gold_pairs)), reverse=True)
        idcg = sum(rel / math.log2(idx + 1) for idx, rel in enumerate(ideal_relevance, start=1))
        ndcg = (dcg / idcg) if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)

    recall_at_k = round(hits_count / float(total_queries), 4)
    mrr_at_k = round(sum(reciprocal_ranks) / float(total_queries), 4)
    ndcg_at_k = round(sum(ndcg_scores) / float(total_queries), 4)
    precision_at_k = round(sum(precisions) / float(total_queries), 4)
    unretrieved = total_queries - hits_count

    return RetrievalMetrics(
        total_queries=total_queries,
        k=top_k,
        recall_at_k=recall_at_k,
        mrr_at_k=mrr_at_k,
        ndcg_at_k=ndcg_at_k,
        precision_at_k=precision_at_k,
        unretrieved_queries=unretrieved
    )

