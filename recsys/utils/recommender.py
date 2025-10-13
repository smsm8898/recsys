import torch
from tqdm import tqdm

def recommend(model, device, test_loader):
    model.to(device)
    model.eval()

    user_recommendations = {}
    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"Recommend ({model.model_name})")
        for sparse_features, dense_features, _ in pbar:
            # test_loader는 user 1명 + 모든 item으로 구성되어 있음
            user_id = sparse_features["user_id"][0].item()
            sparse_features = {k: v.to(device) for k, v in sparse_features.items()}
            dense_features = dense_features.to(device)

            logits = model(sparse_features, dense_features)
            indices = logits.flatten().argsort(descending=True)

            user_recommendations[user_id] = indices.detach().cpu().numpy()
            pbar.set_postfix(user_id=user_id)

    return user_recommendations
