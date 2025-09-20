import numpy as np
from collections import defaultdict
from tqdm import tqdm

def recommend_topk(model, user_seen_movies, topk=10):
    recommend = defaultdict(list)
    # num_users = model.num_users
    scores = model.U @ model.M.T

    for u, seen in user_seen_movies.items():
        # seen = group["movie_id"].unique()
        unseen = np.setdiff1d(np.arange(model.num_movies), seen)
        top_items = np.argsort(scores[u][unseen])[::-1][:topk]
        recommend[u] = unseen[top_items].tolist()
    return recommend

def precision(recommend, test):
    hits = 0
    for row in tqdm(test.itertuples(), total=len(test)):
        u, true_m = row.user_id, row.movie_id
        if true_m in recommend[u]:
            hits += 1
    return hits / len(test)

def ndcg(recommend, test, k=10):
    dcg = 0
    idcg = len(test)  # leave-one-out
    for row in tqdm(test.itertuples(), total=len(test)):
        u, true_m = row.user_id, row.movie_id
        if true_m in recommend[u]:
            rank = recommend[u].index(true_m)
            dcg += 1 / np.log2(rank + 2)
    return dcg / idcg
