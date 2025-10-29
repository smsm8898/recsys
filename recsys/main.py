import os
import json
import yaml
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from recsys.models import MF, NeuMF, FM, WD, DeepFM, DCN, DLRM
from recsys.data.movielens import MovielensDataset
from recsys.utils.metrics import evaluate
from recsys.utils import recommender, visualizer

if __name__ == "__main__":
    # 1. load config
    with open("recsys/configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    with open(config["data"]["num_sparse_features"], "r") as f:
        num_sparse_features = json.load(f)
        
    with open(config["data"]["dense_features"], "r") as f:
        dense_features = json.load(f)

    # 2. Define Dataset
    train = pd.read_parquet(config["data"]["train"])
    test = pd.read_parquet(config["data"]["test"])

    train_ds = MovielensDataset(train, list(num_sparse_features.keys()), dense_features)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
    
    test_ds = MovielensDataset(test, list(num_sparse_features.keys()), dense_features)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=test["item_id"].nunique(), shuffle=False)

    # 3. Define Models
    device = torch.device(config["training"]["device"])
    models = {
        MF.model_name: MF(num_sparse_features={"user_id": num_sparse_features["user_id"], "item_id": num_sparse_features["item_id"]}, **config["models"][MF.model_name]),
        NeuMF.model_name: NeuMF(num_sparse_features={"user_id": num_sparse_features["user_id"], "item_id": num_sparse_features["item_id"]}, **config["models"][NeuMF.model_name]),
        FM.model_name: FM(num_sparse_features=num_sparse_features, dense_features=dense_features, **config["models"][FM.model_name]),
        DeepFM.model_name: DeepFM(num_sparse_features=num_sparse_features, dense_features=dense_features, **config["models"][DeepFM.model_name]),
        DCN.model_name: DCN(num_sparse_features=num_sparse_features, dense_features=dense_features, **config["models"][DCN.model_name]),
        DLRM.model_name: DLRM(num_sparse_features=num_sparse_features, dense_features=dense_features, **config["models"][DLRM.model_name]),
        WD.model_name: WD(num_sparse_features=num_sparse_features, dense_features=dense_features, **config["models"][WD.model_name]),
    }
    optimizers = {}
    for model_name, model in models.items():
        model.to(device)
        optimizers[model_name] = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

    # 4. Train Models
    histories = {}
    for model_name, model in models.items():
        print(model_name)
        optimizer = optimizers[model_name]
        histories[model_name] = recommender.train(model, train_loader, optimizer, device, config["training"]["epochs"])

    # 5. Results Comparison
    fig, ax = plt.subplots(1, 3, figsize=(12, 5), sharex=True)
    for model_name in histories:
        for i, metric_names in enumerate(histories[model_name]):
            ax[i].plot(histories[model_name][metric_names], marker="o", label=model_name)
            ax[i].set_title(metric_names.capitalize())
            ax[i].grid(True, linestyle="--", alpha=0.6)
            ax[i].legend()
    fig.supxlabel("Epoch")
    fig.suptitle(f"{len(models)} Models Train Metrics")
    fig.tight_layout()
    plt.savefig(f"{len(models)} Models Train Metrics")
    plt.show()
    plt.close()
    
    recommendations = {}
    for model_name, model in models.items():
        print(model_name)
        recommendations[model_name] = recommender.inference(model, device, test_loader)
    
    ks = [5, 10, 15, 20]
    _test = test[test["rating"]==1]
    results = evaluate(_test, recommendations, ks)
    visualizer.plot_metrics(results)

        