import os
import numpy as np
import pickle
from collections import defaultdict

class MF:
    def __init__(self, num_users, num_movies, k, lr, reg, eps=1e-9):
        self.num_users = num_users
        self.num_movies = num_movies
        self.k = k
        self.lr = lr
        self.reg = reg
        self.eps = eps
        self.U = np.random.normal(0, 0.1, (num_users, k))
        self.I = np.random.normal(0, 0.1, (num_movies, k))

    def forward(self, uid, iid):
        pred = np.sum(self.U[uid] * self.I[iid], axis=1) # dot product
        pred = 1 / (1 + np.exp(-pred)) # sigmoid
        return pred
    
    def compute_loss(self, uid, iid, label):
        pred = self.forward(uid, iid)
        loss = -(label * np.log(pred + self.eps) + (1-label) * np.log(1 - pred + self.eps))
        return loss.sum()
    
    def update(self, uid, iid, label):
        pred = self.forward(uid, iid)
        error = label - pred
                
        self.U[uid] += self.lr * (error[:, None] * self.I[iid] - self.reg * self.U[uid])
        self.I[iid] += self.lr * (error[:, None] * self.U[uid] - self.reg * self.I[iid])

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/U.npy", self.U)
        np.save(f"{save_dir}/I.npy", self.I)
        with open(f"{save_dir}/model.pkl", "wb") as f:
            pickle.dump(self, f)

    def load(self, save_dir):
        self.U = np.load(f"{save_dir}/U.npy")
        self.I = np.load(f"{save_dir}/I.npy")
        with open(f"{save_dir}/model.pkl", "rb") as f:
            obj = pickle.load(f)
        
        self.__dict__.update(obj.__dict__) # 객체의 다른 속성까지 가져올 수 있음

    def recommend(self, user_seen_items, topk=None):
        recommend = defaultdict(list)
        # num_users = model.num_users
        scores = self.U @ self.I.T # [U, I]

        for u, seen in user_seen_items.items():
            unseen = np.setdiff1d(np.arange(self.num_movies), seen)
            rec = np.argsort(scores[u][unseen])[::-1]
            if topk:
                rec = rec[:topk]
            recommend[u] = unseen[rec].tolist()
        return recommend