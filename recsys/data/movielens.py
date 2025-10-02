import torch

class MovielensDataset(torch.utils.data.Dataset):
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