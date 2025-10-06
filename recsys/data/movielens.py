import torch
import numpy as np

class MovielensSimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data, sparse_feature_names):
        self.sparse_feature_names = sparse_feature_names
        self.data = data[:, :-1]
        self.label = data[:, [-1]].astype(float)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sparse_featues = {
            self.sparse_feature_names[0]: self.data[idx, 0],
            self.sparse_feature_names[1]: self.data[idx, 1],
        }
        labels = self.label[idx]
        return sparse_featues, labels
    
class MovielensFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, df, sparse_feature_names: list[str], dense_feature_names: list[str], label="rating"):
        self.df = df.copy()
        self.sparse_feature_names = sparse_feature_names
        self.dense_feature_names = dense_feature_names
        
        self.sparse_features = {feature_name: self.df[feature_name].values.astype(np.int64) for feature_name in self.sparse_feature_names}
        self.dense_features = self.df[dense_feature_names].values.astype(np.float32)
        self.label = self.df[label].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sparse_features = {k: v[idx] for k, v in self.sparse_features.items()}
        return sparse_features, self.dense_features[idx], self.label[idx]
            
