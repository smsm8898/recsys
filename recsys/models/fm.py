import torch

class FactorizationMachine(torch.nn.Module):
    def __init__(
        self,
        num_sparse_features: int,
        num_dense_features: int,
        latent_dim: int
    ):
        super().__init__()
        self.num_sparse_features = num_sparse_features
        self.num_dense_features = num_dense_features
        self.latent_dim = latent_dim

        self.W0 = torch.nn.Parameter(torch.zeros(1)) # global bias
        
        self.W_sparse = torch.nn.Embedding(num_sparse_features, 1)
        self.W_dense = torch.nn.Linear(num_dense_features, 1)
        
        self.V_sparse = torch.nn.Embedding(num_sparse_features, latent_dim)
        self.V_dense = torch.nn.Linear(num_dense_features, latent_dim)

    def forward(self, sparse_features: torch.LongTensor, dense_features: torch.FloatTensor) -> torch.FloatTensor:
        first_sparse_term = self.W_sparse(sparse_features).sum(dim=1) # [batch_size, 1]
        first_dense_term = self.W_dense(dense_features) # [batch_size, 1]
        first_term = first_sparse_term + first_dense_term # [batch_size, 1]
        
        v_sparse = self.V_sparse(sparse_features) # [batch_size, num_sparse_fields, latent_dim]
        v_dense = self.V_dense(dense_features).unsqueeze(1) # [batch_size, 1, latent_dim]
        v = torch.cat([v_sparse, v_dense], dim=1) # [batch_size, num_sparse_fields + 1, latent_dim] 
        
        sum_of_square = v.sum(dim=1).pow(2) # [batch_size, latent_dim]
        squares_of_sum = v.pow(2).sum(dim=1) # [batch_size, latent_dim]
        second_term = 0.5 * (sum_of_square - squares_of_sum).sum(dim=1, keepdim=True)

        logits = self.W0 + first_term + second_term
        return logits