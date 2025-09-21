import numpy as np
import pandas as pd
from tqdm import tqdm

def leave_one_out_split(df, column="user_id"):
    """Split train, test by timestamp"""
    train, test = [], []
    df = df.sort_values(by="timestamp")
    for u, group in df.groupby(column):
        train.append(group.iloc[:-1].copy())
        test.append(group.iloc[[-1]].copy())

    train = pd.concat(train, axis=0)
    test = pd.concat(test, axis=0)
    return train, test

def random_negative_sampling(df, num_negative_sample, num_items, name_item="movie_id"):
    """Generate Random Negative Sample"""
    negative_sample = []
    for _ in range(num_negative_sample):
        _neg = df.copy()
        _neg[name_item] = np.random.randint(low=0, high=num_items, size=len(df))
        negative_sample.append(_neg)

    negative_sample = pd.concat(negative_sample, axis=0)
    negative_sample["label"] = 0
    train_with_neg = pd.concat([df.copy(), negative_sample], axis=0).drop_duplicates(keep="first")
    return train_with_neg

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