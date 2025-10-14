import torch

class FactorizationMachine(torch.nn.Module):
    model_name = "fm"
    def __init__(self, num_sparse_features: dict[str, int], dense_features: list[str], latent_dim: int):
        super().__init__()
        self.num_sparse_features = num_sparse_features
        self.sparse_feature_names = list(num_sparse_features.keys()) # F
        self.dense_feature_names = dense_features
        self.num_dense_features = len(dense_features)
        self.latent_dim = latent_dim
        

        # bias
        self.bias = torch.nn.Parameter(torch.zeros(1, 1))

        # 1nd: linear
        self.linear_sparse = torch.nn.ModuleDict({
            name: torch.nn.Embedding(num_sparse_feature, 1)
            for name, num_sparse_feature in num_sparse_features.items()
        })
        self.linear_dense = torch.nn.Linear(self.num_dense_features, 1)
        
        # 2nd: cross
        self.sparse_arch = torch.nn.ModuleDict({
            name: torch.nn.Embedding(num_sparse_feature, latent_dim)
            for name, num_sparse_feature in num_sparse_features.items()
        })
        self.dense_arch = torch.nn.Linear(self.num_dense_features, latent_dim)


    def forward(self, features: dict[str, torch.Tensor]) -> torch.FloatTensor:
        dense_features = torch.stack([features[feature_name] for feature_name in self.dense_feature_names], dim=1)

        # 1st: linear
        first_term = [self.linear_dense(dense_features)]
        for feature_name in self.sparse_feature_names:
            # [B, (F+1), 1]
            first_term.append(self.linear_sparse[feature_name](features[feature_name])) 
        first_term = torch.cat(first_term, dim=1).sum(dim=1, keepdim=True) # [B, 1]
        
        
        # 2nd: cross
        second_term = [self.dense_arch(dense_features)]
        for feature_name in self.sparse_feature_names:
            # [B, (F+1), D]
            second_term.append(self.sparse_arch[feature_name](features[feature_name]))
        second_term = torch.stack(second_term, dim=1) 
        
        sum_of_square = second_term.sum(dim=1).pow(2) # [B, D]
        squares_of_sum = second_term.pow(2).sum(dim=1) # [B, D]
        second_term = 0.5 * (sum_of_square - squares_of_sum).sum(dim=1, keepdim=True) # [B, 1]

        
        # final
        logits = first_term + second_term + self.bias
        return logits