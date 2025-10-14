from collections import defaultdict
import numpy as np


def evaluate(test, recommendations, ks, controls=["popular", "random"]):
    result = defaultdict(float)
    for line in test.itertuples():
        user_id, item_id = line.user_id, line.item_id
    
        for rec_method in recommendations:
            if rec_method in controls:
                recs = recommendations[rec_method]
            else:
                recs = recommendations[rec_method][user_id]
    
            for k in ks:
                result[f"hr_{rec_method}_{k}"] += hit_ratio_at_k(recs, item_id, k)
                result[f"ndcg_{rec_method}_{k}"] += ndcg_at_k(recs, item_id, k)
                result[f"mrr_{rec_method}_{k}"] += mrr_at_k(recs, item_id, k)
    return {k: np.round(v/len(test), 4) for k, v in result.items()}


# hr
def hit_ratio_at_k(recs: np.ndarray, real: int, k: int) -> float:
    return 1.0 if real in recs[:k] else 0.0


# ndcg
def ndcg_at_k(recs: np.ndarray, real: int, k: int) -> float:
    recs_k = recs[:k]
    idx = np.where(recs_k == real)[0]
    if len(idx) > 0:
        rank = idx[0] + 1  # 1-based
        return 1.0 / np.log2(rank + 1)
    return 0.0


# mrr
def mrr_at_k(recs: np.ndarray, real: int, k: int) -> float:
    recs_k = recs[:k]
    idx = np.where(recs_k == real)[0]
    if len(idx) > 0:
        rank = idx[0] + 1
        return 1.0 / rank
    return 0.0
