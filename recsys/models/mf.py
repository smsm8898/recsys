import torch

class MatrixFactorization(torch.nn.Module):
    def __init__(
        self,
        num_sparse_features: dict[str, int],
        latent_dim: int,
    ):
        super().__init__()
        self.num_sparse_features = num_sparse_features
        self.latent_dim = latent_dim
        self.sparse_feature_names = list(num_sparse_features.keys())

        self.embeddings = torch.nn.ModuleDict({
            name : torch.nn.Embedding(num_sparse_feature, latent_dim)
            for name, num_sparse_feature in num_sparse_features.items()
        })
        
    def forward(self, sparse_features: dict[str, torch.LongTensor]) -> torch.FloatTensor:
        emb1 = self.embeddings[self.sparse_feature_names[0]](sparse_features[self.sparse_feature_names[0]])
        emb2 = self.embeddings[self.sparse_feature_names[1]](sparse_features[self.sparse_feature_names[1]])
        logits = (emb1 * emb2).sum(dim=1, keepdim=True)
        return logits