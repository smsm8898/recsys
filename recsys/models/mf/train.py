import os
import json
import matplotlib.pyplot as plt
import argparse
import numpy as np
from recsys.data.movielens import MOVIELENS100K
from recsys.data.utils import leave_one_out_split, random_negative_sampling, precision, ndcg
from recsys.models.mf.model import MF

def get_args():
    parser = argparse.ArgumentParser(description="Matrix Factorization Model Training")
    parser.add_argument("--base_path", type=str, help="프로젝트의 기본 경로")
    parser.add_argument("--data_name", type=str, default="movielens", help="데이터셋 이름")
    parser.add_argument("--random_seed", type=int, default=42, help="랜덤 값 고정")
    parser.add_argument("--k", type=int, default=16, help="잠재 요인(latent factor)의 수")
    parser.add_argument("--epochs", type=int, default=10, help="학습 에포크 수")
    parser.add_argument("--lr", type=float, default=0.01, help="학습률(Learning Rate)")
    parser.add_argument("--reg", type=float, default=0.01, help="정규화(regularization) 값")
    parser.add_argument("--batch_size", type=int, default=16, help="배치 사이즈")
    parser.add_argument("--num_negative_sample", type=int, default=5, help="부정적 샘플링 수")
    parser.add_argument("--topk", type=int, default=10, help="추천 목록의 상위 K개")
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = get_args()
    np.random.seed = args.random_seed

    history = {
        "loss": [],
        "random_seed": args.random_seed,
        "k": args.k,
        "epochs": args.epochs,
        "lr": args.lr,
        "reg": args.reg,
        "batch_size": args.batch_size,
        "num_negative_sample": args.num_negative_sample,
        "topk": args.topk,
    }

    # 데이터 불러오기 & 전처리
    data_path = os.path.join(args.base_path, "data", args.data_name)
    output_path = os.path.join(args.base_path, "outputs", args.data_name, "mf")
    os.makedirs(output_path, exist_ok=True)

    movielens = MOVIELENS100K(path=data_path)
    ratings, users, movies = movielens.load()
    ratings = movielens.encode_index(ratings, users, movies) # id mapping
    num_users, num_movies = len(users), len(movies)
    train, test = leave_one_out_split(ratings)
    train_with_neg = random_negative_sampling(train, num_negative_sample=args.num_negative_sample, num_items=num_movies)


    # 모델 초기화
    model = MF(num_users=num_users, num_movies=num_movies, k=args.k, lr=args.lr, reg=args.reg)

    # 학습
    for epoch in range(args.epochs):
        train_with_neg = train_with_neg.sample(frac=1)  # random shuffle
        total_loss = 0

        for i in range(0, len(train_with_neg), args.batch_size):
            batch = train_with_neg.iloc[i:i+args.batch_size]
            u, m, l = batch["user_id"].values, batch["movie_id"].values, batch["label"].values
            total_loss +=  model.compute_loss(u, m, l)
            model.update(u, m, l)

        avg_loss = total_loss / len(train_with_neg)
        history["loss"].append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    
    # 추천 평가
    print("추천 평가 시작...")
    user_seen_movies = train.groupby("user_id")["movie_id"].unique()
    rec_list = model.recommend(user_seen_movies, topk=args.topk)
    _precision = precision(recommend=rec_list, test=test)
    _ndcg = ndcg(recommend=rec_list, test=test)
    history[f"precision@{args.topk}"] = _precision
    history[f"ndcg@{args.topk}"] = _ndcg
    print(f"Precision@{args.topk}: {_precision:.4f}, NDCG@{args.topk}: {_ndcg:.4f}")

    # 저장
    model.save(output_path)

    with open(os.path.join(output_path, "history.json"), "w") as f:
        json.dump(history, f, indent=4)

    with open(os.path.join(output_path, "recommendations.json"), "w") as f:
        json.dump(rec_list, f)