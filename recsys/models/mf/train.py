import json
import os
import matplotlib.pyplot as plt
from recsys.models.mf.movielens_loader import load_movielens, leave_one_out_split, random_negative_sampling, save_index
from recsys.models.mf.model import MF
from recsys.models.mf.evaluate import recommend_topk, precision, ndcg

# -----------------------------
# 1. 설정
# -----------------------------
root_dir = "/Users/seungminjang/Desktop/workspace/recsys/recsys"
data_name = "movielens"
model_name = "mf"


rating_path = os.path.join(root_dir, f"data/{data_name}/u.data")
user_path = os.path.join(root_dir, f"data/{data_name}/u.user")
movie_path = os.path.join(root_dir, f"data/{data_name}/u.item")
outputs_dir = os.path.join(root_dir, f"outputs/{data_name}/{model_name}")
os.makedirs(outputs_dir, exist_ok=True)


k = 16
epochs = 10
lr = 0.01
reg = 0.01
batch_size = 16
num_negative_sample = 5
topk = 10

# -----------------------------
# 2. 데이터 불러오기 & 전처리
# -----------------------------
ratings, user_to_idx, movie_to_idx = load_movielens(rating_path, user_path, movie_path)
num_users, num_movies = len(user_to_idx), len(movie_to_idx)

train, test = leave_one_out_split(ratings)
train_with_neg = random_negative_sampling(train, num_movies, num_negative_sample=num_negative_sample)

# -----------------------------
# 3. 모델 초기화
# -----------------------------
mf_model = MF(num_users=num_users, num_movies=num_movies, k=k, lr=lr, reg=reg)

# -----------------------------
# 4. 학습
# -----------------------------
history = {"loss": []}

for epoch in range(epochs):
    train_with_neg = train_with_neg.sample(frac=1)  # random shuffle
    total_loss = 0

    for i in range(0, len(train_with_neg), batch_size):
        batch = train_with_neg.iloc[i:i+batch_size]
        u, m, c = batch["user_id"].values, batch["movie_id"].values, batch["click"].values
        total_loss +=  mf_model.compute_loss(u, m, c)
        mf_model.update(u, m, c)

    avg_loss = total_loss / len(train_with_neg)
    history["loss"].append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# -----------------------------
# 5. 학습 결과 시각화
# -----------------------------
plt.plot(history["loss"])
plt.xlabel("Epoch")
plt.ylabel("Binary Cross Entropy Loss")
plt.title("MF Train Loss")
plt.tight_layout()

# PNG로 저장
plt.savefig(f"{outputs_dir}/train_loss.png")
print(f"학습 loss 그래프 saved: {outputs_dir}/train_loss.png")

# -----------------------------
# 6. 추천 평가
# -----------------------------
print("추천 평가 시작...")
user_seen_movies = train.groupby("user_id")["movie_id"].unique()
rec_list = recommend_topk(mf_model, user_seen_movies, topk=topk)
_precision = precision(recommend=rec_list, test=test)
_ndcg = ndcg(recommend=rec_list, test=test)
print(f"Precision@{topk}: {_precision:.4f}, NDCG@{topk}: {_ndcg:.4f}")


# -----------------------------
# 7. 모델 및 인덱스 저장
# -----------------------------
mf_model.save(outputs_dir)
save_index(user_to_idx, movie_to_idx, outputs_dir)
with open(f"{outputs_dir}/recommendations.json", "w") as f:
    json.dump(rec_list, f)
print(f"모델 saved: {outputs_dir}/mf_model.pkl")
print("user_to_index, movie_to_index saved.")

