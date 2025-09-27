import torch

class MatrixFactorization(torch.nn.Module):
    def __init__(
        self,
        num_sparse_features: int,
        latent_dim: int,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.W = torch.nn.Embedding(num_sparse_features, latent_dim)

    def forward(self, sparse_features: torch.LongTensor) -> torch.FloatTensor:
        embeddings = self.W(sparse_features) # [batch_size, 2, latent_dim]
        logits = (embeddings[:, 0, :] * embeddings[:, 1, :]).sum(dim=1, keepdim=True) # [batch_size, 1]
        return logits