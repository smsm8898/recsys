from collections import defaultdict
import os
import numpy as np
import torch
from tqdm import tqdm

class GMF(torch.nn.Module):
    def __init__(self, num_users: int, num_items: int, k: int):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(num_users, k)
        self.item_embedding = torch.nn.Embedding(num_items, k)

    def forward(self, uid: torch.LongTensor, iid: torch.LongTensor) -> torch.FloatTensor:
        user_emb = self.user_embedding(uid)
        item_emb = self.item_embedding(iid)
        out = user_emb * item_emb
        return out

class MLP(torch.nn.Module):
    def __init__(self, num_users: int, num_items: int, k: int, hidden_layers: list[int]):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(num_users, k)
        self.item_embedding = torch.nn.Embedding(num_items, k)

        mlp = []
        in_features = 2 * k # Concatenate user and item embeddings
        for out_features in hidden_layers:
            mlp.append(torch.nn.Linear(in_features=in_features, out_features=out_features))
            mlp.append(torch.nn.ReLU())
            mlp.append(torch.nn.Dropout(0.2))
            in_features = out_features
        self.mlp = torch.nn.ModuleList(mlp)

    def forward(self, uid: torch.LongTensor, iid: torch.LongTensor) -> torch.FloatTensor:
        user_emb = self.user_embedding(uid)
        item_emb = self.item_embedding(iid)
        out = torch.cat([user_emb, item_emb], dim=-1)
        for m in self.mlp:
            out = m(out)
        return out


class NCF(torch.nn.Module):
    def __init__(self, num_users: int, num_items: int, k: int, hidden_layers: list[int], lr: float, reg: float):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.k = k
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.reg = reg

        
        self.gmf = GMF(num_users, num_items, k)
        self.mlp = MLP(num_users, num_items, k, hidden_layers)

        in_features = k + hidden_layers[-1]
        self.classifier = torch.nn.Linear(in_features=in_features, out_features=1)

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=reg)

    def forward(self, uid: torch.LongTensor, iid: torch.LongTensor) -> torch.FloatTensor:
        gmf_out = self.gmf(uid, iid)
        mlp_out = self.mlp(uid, iid)
        
        out = torch.cat([gmf_out, mlp_out], dim=-1)
        logits = self.classifier(out)
        return logits
    
    def compute_loss(self, uid, iid, label):
        logits = self.forward(uid, iid) 
        loss = self.criterion(logits, label)
        return loss
    
    def update(self, uid, iid, label):
        self.optimizer.zero_grad()
        loss = self.compute_loss(uid, iid, label)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, "model.pt"))

    def load(self, save_dir):
        self.load_state_dict(torch.load(os.path.join(save_dir, "model.pt")))
    
    def recommend(self, user_seen_items: dict[int, list[int]], topk: int):
        self.eval()
        recommend = defaultdict()
        all_items = np.arange(self.num_items)

        with torch.no_grad():
            for u, seen in tqdm(user_seen_items.items(), total=len(user_seen_items)):
                unseen = np.setdiff1d(all_items, seen)

                iid = torch.LongTensor(unseen)
                uid = torch.LongTensor([u] * len(unseen))

                logits = self.forward(uid, iid).squeeze()
                _, indices = torch.topk(logits, k=topk)
                recommend[u] = iid[indices].detach().cpu().numpy().tolist()
        return recommend
