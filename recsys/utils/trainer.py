import torch
import numpy as np

from sklearn.metrics import roc_auc_score
from collections import defaultdict
from tqdm import tqdm


def train(model, train_loader, optimizer, device, epochs):
    history = defaultdict(list)
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        all_labels = []
        all_preds = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for sparse_features, dense_features, labels in pbar:
            # Move data to device
            sparse_features = {k: v.to(device) for k, v in sparse_features.items()}
            dense_features = dense_features.to(device)
            labels = labels.unsqueeze(1).to(device)

            # Forward pass
            logits = model(sparse_features, dense_features)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            total_loss += loss.item()
            probs = logits.sigmoid().detach().cpu().numpy()
            all_preds.extend(probs.flatten())
            all_labels.extend(labels.detach().cpu().numpy().flatten())
            
            pbar.set_postfix(logloss=f"{loss.item():.4f}")

        # Epoch-level metrics
        avg_loss = total_loss / len(train_loader)
        acc = ((np.array(all_preds) > 0.5) == np.array(all_labels)).mean()
        auc = roc_auc_score(all_labels, all_preds)

        history["accuracy"].append(acc)
        history["auroc"].append(auc)
        history["logloss"].append(avg_loss)
        
        # print(f"Epoch {epoch+1:02d} | LogLoss: {avg_loss:.4f} | AUC: {auc:.4f} | ACC: {acc:.4f}")
    
    return history