import torch
import numpy as np

    
class MovielensDataset(torch.utils.data.Dataset):
    def __init__(self, df, sparse_feature_names, dense_feature_names=None, label:str="rating"):
        self.df = df.copy()
        
        self.label = self.df[label].astype(np.float32)
        
        self.features = {}
        for feature_name in sparse_feature_names:
            self.features[feature_name] = self.df[feature_name].values.astype(np.int64)

        if dense_feature_names:
            for feature_name in dense_feature_names:
                self.features[feature_name] = self.df[feature_name].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        feature = {k: v[idx] for k, v in self.features.items()}
        label = self.label[idx]
        return feature, label