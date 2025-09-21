import os
import requests
import pandas as pd
from tqdm import tqdm

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
URL = "http://files.grouplens.org/datasets/movielens/ml-100k"

class MOVIELENS100K:
    def __init__(self, path: str):
        self.file_names = ["u.data", "u.item", "u.user"]
        if path.split(os.path.sep)[-1] != "movielens":
            path = os.path.join(path, "movielens")
        self.path = path
        os.makedirs(self.path, exist_ok=True)

    def download(self):
        """Download Movielens-100k"""
        for file_name in tqdm(self.file_names, desc="Download MovieLens-100k"):
            file_path = os.path.join(self.path, file_name)

            if os.path.exists(file_path):
                print(f"File already exists: {file_name}. Skipping download.")
                continue # 다운로드를 건너뛰고 다음 파일로 이동

            print(f"Downloading {file_name}...")
            try:
                with requests.get(f"{URL}/{file_name}", stream=True) as r:
                    r.raise_for_status()
                    with open(file_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                print(f"Successfully downloaded {file_name}")
            except requests.exceptions.RequestException as e:
                print(f"Error downloading {file_name}: {e}")

    def load(self, is_implicit=True):
        """Read Movielens-100k data"""
        ratings = pd.read_csv(self.path + "/u.data", sep="\t", names=COLUMNS["ratings"])
        users = pd.read_csv(self.path + "/u.user", sep="|", names=COLUMNS["users"], encoding="latin-1")
        movies = pd.read_csv(self.path + "/u.item", sep="|", names=COLUMNS["movies"], encoding="latin-1")

        # Ratings -> Implicit feedback
        if is_implicit:
            ratings = ratings[ratings["rating"] >= 4]
            ratings["label"] = 1

        return ratings, users, movies
    
    def encode_index(self, ratings, users=None, movies=None):
        if users is not None:
            user_to_idx = {uid:i for i, uid in enumerate(sorted(users["user_id"]))}
            ratings["user_id"] = ratings["user_id"].map(user_to_idx)
        if movies is not None:
            movie_to_idx = {mid:j for j, mid in enumerate(sorted(movies["movie_id"]))}
            ratings["movie_id"] = ratings["movie_id"].map(movie_to_idx)
        return ratings
    


# def save_index(user_to_idx, movie_to_idx, save_dir):
#     os.makedirs(save_dir, exist_ok=True)
#     with open(f"{save_dir}/user_to_idx.json", "w") as f:
#         json.dump(user_to_idx, f)
#     with open(f"{save_dir}/movie_to_idx.json", "w") as f:
#         json.dump(movie_to_idx, f)

# def load_index(save_dir):
#     with open(f"{save_dir}/user_to_idx.json", "r") as f:
#         user_to_idx = json.load(f)
#     with open(f"{save_dir}/movie_to_idx.json", "r") as f:
#         movie_to_idx = json.load(f)
#     return user_to_idx, movie_to_idx