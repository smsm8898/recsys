import torch
import numpy as np

from sklearn.metrics import roc_auc_score
from collections import defaultdict
from tqdm import tqdm


def train(model, train_loader, optimizer, device, epochs):
    history = defaultdict(list)
    model.to(device)
    model.train()
    auc = 0
    for epoch in range(epochs):
        total_loss = 0
        all_labels = []
        all_preds = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for features, labels in pbar:
            # To device
            features = {k: v.to(device) for k, v in features.items()}
            labels = labels.unsqueeze(1).to(device)

            # Forward
            logits = model(features)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            total_loss += loss.item()
            probs = logits.sigmoid().detach().cpu().numpy()
            all_preds.extend(probs.flatten())
            all_labels.extend(labels.detach().cpu().numpy().flatten())

            # tqdm update
            pbar.set_postfix(
                logloss=loss.item(),
                auc=f"{auc:.4f}"
            )

        
        avg_loss = total_loss / len(train_loader)
        acc = ((np.array(all_preds) > 0.5) == np.array(all_labels)).mean()
        auc = roc_auc_score(all_labels, all_preds)
        history["accuracy"].append(acc)
        history["auroc"].append(auc)
        history["logloss"].append(avg_loss)
        
        pbar.set_postfix(
            logloss=f"{avg_loss:.4f}",
            auc=f"{auc:.4f}"
        )
        pbar.refresh()
    return history


def inference(model, device, test_loader):
    model.to(device)
    model.eval()
    user_recommendations = {}
    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"Recommend ({model.model_name})")
        for features, _ in pbar:
            # test_loader는 user 1명 + 모든 item으로 구성되어 있음
            user_id = features["user_id"][0].item()
            features = {k: v.to(device) for k, v in features.items()}
            logits = model(features)
            indices = logits.flatten().argsort(descending=True)
            user_recommendations[user_id] = indices.detach().cpu().numpy()
            pbar.set_postfix(user_id=user_id)
    return user_recommendations