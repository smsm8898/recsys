import numpy as np
import pickle
import os

class MF:
    def __init__(self, num_users, num_movies, k, lr, reg, eps=1e-9):
        self.num_users = num_users
        self.num_movies = num_movies
        self.k = k
        self.lr = lr
        self.reg = reg
        self.eps = eps
        self.U = np.random.normal(0, 0.1, (num_users, k))
        self.M = np.random.normal(0, 0.1, (num_movies, k))

    def forward(self, u_idx, m_idx):
        pred = np.sum(self.U[u_idx] * self.M[m_idx], axis=1) # dot product
        pred = 1 / (1 + np.exp(-pred)) # sigmoid
        return pred
    
    def compute_loss(self, u_idx, m_idx, click):
        pred = self.forward(u_idx, m_idx)
        loss = -(click * np.log(pred + self.eps) + (1-click) * np.log(1 - pred + self.eps))
        return loss.sum()
    
    def update(self, u_idx, m_idx, click):
        pred = self.forward(u_idx, m_idx)
        error = click - pred
                
        self.U[u_idx] += self.lr * (error[:, None] * self.M[m_idx] - self.reg * self.U[u_idx])
        self.M[m_idx] += self.lr * (error[:, None] * self.U[u_idx] - self.reg * self.M[m_idx])


    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/U.npy", self.U)
        np.save(f"{save_dir}/M.npy", self.M)
        with open(f"{save_dir}/mf_model.pkl", "wb") as f:
            pickle.dump(self, f)

    def load(self, save_dir):
        self.U = np.load(f"{save_dir}/U.npy")
        self.M = np.load(f"{save_dir}/M.npy")
        with open(f"{save_dir}/mf_model.pkl", "rb") as f:
            obj = pickle.load(f)
        
        self.__dict__.update(obj.__dict__) # 객체의 다른 속성까지 가져올 수 있음
