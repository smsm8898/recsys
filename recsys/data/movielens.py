import os
import pandas as pd


_COLUMNS = {
    "ratings": ["user_id", "item_id", "rating", "timestamp"],
    "users": ["user_id", "age", "gender", "occupation", "zip"],
    "items": [
        "item_id",
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

# "http://files.grouplens.org/datasets/movielens/ml-100k"
def movielens100k(path):
    # Implicit Feedback
    feedback = pd.read_csv(os.path.join(path, "u.data"), sep="\t", names=_COLUMNS["ratings"])
    users = pd.read_csv(os.path.join(path, "u.user"), sep="|", names=_COLUMNS["users"], encoding="latin-1")
    items = pd.read_csv(os.path.join(path, "u.item"), sep="|", names=_COLUMNS["items"], encoding="latin-1")
    feedback["label"] = 1
    return feedback, users, items