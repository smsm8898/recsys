import numpy as np

def hit_ratio_at_k(recs: np.ndarray, real: int, k: int) -> float:
    return 1.0 if real in recs[:k] else 0.0


def ndcg_at_k(recs: np.ndarray, real: int, k: int) -> float:
    recs_k = recs[:k]
    idx = np.where(recs_k == real)[0]
    if len(idx) > 0:
        rank = idx[0] + 1  # 1-based
        return 1.0 / np.log2(rank + 1)
    return 0.0


def mrr_at_k(recs: np.ndarray, real: int, k: int) -> float:
    recs_k = recs[:k]
    idx = np.where(recs_k == real)[0]
    if len(idx) > 0:
        rank = idx[0] + 1
        return 1.0 / rank
    return 0.0
