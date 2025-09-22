import os
import json
import torch
import argparse
from tqdm import tqdm

from recsys.data.movielens import MOVIELENS100K
from recsys.data.utils import leave_one_out_split, random_negative_sampling, precision, ndcg
from recsys.models.ncf.model import NCF


class MovielensDataset(torch.utils.data.Dataset):
    def __init__(self, ratings):
        self.users = ratings["user_id"].astype(int).values
        self.items = ratings["movie_id"].astype(int).values
        self.labels = ratings[["label"]].astype(float).values

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

def get_args():
    parser = argparse.ArgumentParser(description="Matrix Factorization Model Training")
    parser.add_argument("--base_path", type=str, help="프로젝트의 기본 경로")
    parser.add_argument("--data_name", type=str, default="movielens", help="데이터셋 이름")
    parser.add_argument("--random_seed", type=int, default=42, help="랜덤 값 고정")
    parser.add_argument("--k", type=int, default=16, help="잠재 요인(latent factor)의 수")
    parser.add_argument("--hidden_layers", type=str, default="32,16,8", help="MLP 은닉층 크기 (쉼표로 구분)")
    parser.add_argument("--epochs", type=int, default=10, help="학습 에포크 수")
    parser.add_argument("--lr", type=float, default=0.01, help="학습률(Learning Rate)")
    parser.add_argument("--reg", type=float, default=0.01, help="정규화(regularization) 값")
    parser.add_argument("--batch_size", type=int, default=16, help="배치 사이즈")
    parser.add_argument("--num_negative_sample", type=int, default=5, help="부정적 샘플링 수")
    parser.add_argument("--topk", type=int, default=10, help="추천 목록의 상위 K개")
    args = parser.parse_args()
    return args


def train_model(model, train_loader, epochs, history):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        
        for step, (users, items, labels) in pbar:
            loss = model.update(users, items, labels)
            total_loss += loss
            pbar.set_postfix(loss=loss)
            
        avg_loss = total_loss / len(train_loader)
        history["loss"].append(avg_loss)
        tqdm.write(f"Epoch {epoch+1} finished, Avg Loss: {avg_loss:.4f}")


def main():
    args = get_args()

    torch.manual_seed(args.random_seed)
    # np.random.seed(args.random_seed)
    hidden_layers = [int(x) for x in args.hidden_layers.split(',')]

    history = {
        "loss": [],
        "random_seed": args.random_seed,
        "k": args.k,
        "hidden_layers": hidden_layers,
        "epochs": args.epochs,
        "lr": args.lr,
        "reg": args.reg,
        "batch_size": args.batch_size,
        "num_negative_sample": args.num_negative_sample,
        "topk": args.topk,
    }

    
    # 데이터 불러오기 & 전처리
    data_path = os.path.join(args.base_path, "data", args.data_name)
    output_path = os.path.join(args.base_path, "outputs", "ncf", args.data_name)
    os.makedirs(output_path, exist_ok=True)

    movielens = MOVIELENS100K(path=data_path)
    ratings, users, movies = movielens.load()
    ratings = movielens.encode_index(ratings, users, movies)
    num_users, num_movies = len(users), len(movies)
    train, test = leave_one_out_split(ratings, random_state=args.random_seed)
    train_with_neg = random_negative_sampling(
        train,
        num_negative_sample=args.num_negative_sample,
        num_items=num_movies,
        random_state=args.random_seed
    )
    
    # 모델 초기화
    model = NCF(
        num_users=num_users,
        num_items=num_movies,
        k=args.k,
        hidden_layers=hidden_layers,
        lr=args.lr,
        reg=args.reg
    )

    
    
    # PyTorch DataLoader 설정
    train_dataset = MovielensDataset(train_with_neg)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # 모델 학습
    train_model(model, train_loader, args.epochs, history)

    
     # 추천 평가
    print("추천 평가 시작...")
    user_seen_movies = train.groupby("user_id")["movie_id"].unique()
    rec_list = model.recommend(user_seen_movies, topk=args.topk)
    _precision = precision(recommend=rec_list, test=test)
    _ndcg = ndcg(recommend=rec_list, test=test)
    history[f"precision@{args.topk}"] = _precision
    history[f"ndcg@{args.topk}"] = _ndcg
    print(f"Precision@{args.topk}: {_precision:.4f}, NDCG@{args.topk}: {_ndcg:.4f}")

if __name__ == "__main__":
    main()