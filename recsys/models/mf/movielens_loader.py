import os
import json
import pandas as pd
import numpy as np


COLUMNS = {
    "ratings": ["user_id", "movie_id", "rating", "timestamp"],
    "users": ["user_id", "age", "gender", "occupation", "zip"],
    "movies": [
        "movie_id",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url",
        "unknown",
        "action",
        "adventure",
        "animation",
        "children",
        "comedy",
        "crime",
        "documentary", 
        "drama", 
        "fantasy",
        "film-noir",
        "horror",
        "musical",
        "mystery",
        "romance",
        "sci-Fi",
        "thriller",
        "war",
        "western",
    ]
}


def load_movielens(rating_path, user_path, movie_path):
    """Movie Lens의 ratings 불러오기"""
    ratings = pd.read_csv(rating_path, sep="\t", names=COLUMNS["ratings"])
    users = pd.read_csv(user_path, sep="|", names=COLUMNS["users"], encoding="latin-1")
    movies = pd.read_csv(movie_path, sep="|", names=COLUMNS["movies"], encoding="latin-1")

    # Ratings -> Implicit feedback
    ratings = ratings[ratings["rating"] >= 4]
    ratings["click"] = 1
    
    # id mapping
    user_to_idx = {uid:i for i, uid in enumerate(sorted(users["user_id"]))}
    movie_to_idx = {mid:j for j, mid in enumerate(sorted(movies["movie_id"]))}

    ratings["user_id"] = ratings["user_id"].map(user_to_idx)
    ratings["movie_id"] = ratings["movie_id"].map(movie_to_idx)

    return ratings, user_to_idx, movie_to_idx

def leave_one_out_split(ratings):
    """Split train, test by timestamp"""
    train, test = [], []
    ratings = ratings.sort_values(by="timestamp")
    for u, group in ratings.groupby("user_id"):
        train.append(group.iloc[:-1].copy())
        test.append(group.iloc[[-1]].copy())

    train = pd.concat(train, axis=0).drop(["rating", "timestamp"], axis=1)
    test = pd.concat(test, axis=0).drop(["rating", "timestamp"], axis=1)
    return train, test

def random_negative_sampling(train, num_movies, num_negative_sample):
    """Generate Random Negative Sample"""
    negative_sample = []
    for _ in range(num_negative_sample):
        _neg = train.copy()
        _neg["movie_id"] = np.random.randint(low=0, high=num_movies, size=len(train))
        negative_sample.append(_neg)

    negative_sample = pd.concat(negative_sample, axis=0)
    negative_sample["click"] = 0
    train_with_neg = pd.concat([train, negative_sample], axis=0).drop_duplicates(keep="first")
    return train_with_neg

def save_index(user_to_idx, movie_to_idx, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/user_to_idx.json", "w") as f:
        json.dump(user_to_idx, f)
    with open(f"{save_dir}/movie_to_idx.json", "w") as f:
        json.dump(movie_to_idx, f)

def load_index(save_dir):
    with open(f"{save_dir}/user_to_idx.json", "r") as f:
        user_to_idx = json.load(f)
    with open(f"{save_dir}/movie_to_idx.json", "r") as f:
        movie_to_idx = json.load(f)
    return user_to_idx, movie_to_idx