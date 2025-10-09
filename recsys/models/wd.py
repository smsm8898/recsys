import torch

class WideAndDeep(torch.nn.Module):
    model_name = "wd"
    def __init__(
        self,
        num_sparse_features: dict[str, int], 
        num_dense_features: int, 
        latent_dim: int,
        hidden_layers: list[int],
        *args,
        **kwargs,
    ):
        super().__init__()
        self.num_sparse_features = num_sparse_features
        self.num_dense_features = num_dense_features
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.sparse_feature_names = list(num_sparse_features.keys()) # F
        self.F = len(self.sparse_feature_names)

        # Bias
        self.bias = torch.nn.Parameter(torch.zeros(1, 1))

        # Wide: linear
        self.wide_sparse = torch.nn.ModuleDict({
            name: torch.nn.Embedding(num_sparse_feature, 1)
            for name, num_sparse_feature in num_sparse_features.items()
        })
        self.wide_dense = torch.nn.Linear(num_dense_features, 1)

        # Deep
        self.deep_sparse = torch.nn.ModuleDict({
            name: torch.nn.Embedding(num_sparse_feature, latent_dim)
            for name, num_sparse_feature in num_sparse_features.items()
        })
        self.deep_dense = torch.nn.Linear(num_dense_features, latent_dim)
        
        mlp = []
        in_features = (self.F + 1) * latent_dim
        for out_features in hidden_layers:
            mlp.append(torch.nn.Linear(in_features, out_features))
            mlp.append(torch.nn.ReLU())
            in_features = out_features
        if hidden_layers[-1] != 1:
            mlp.append(torch.nn.Linear(in_features, 1))
        self.mlp = torch.nn.Sequential(*mlp)


    def forward(self, sparse_features: dict[str, torch.LongTensor], dense_features: torch.FloatTensor) -> torch.FloatTensor:
        # Wide
        wide_out = [self.wide_dense(dense_features)]
        for name in self.sparse_feature_names:
            wide_out.append(self.wide_sparse[name](sparse_features[name]))
        wide_out = torch.cat(wide_out, dim=1).sum(dim=1, keepdim=True) # [B, 1]
        
        # Deep
        deep_out = [self.deep_dense(dense_features)]
        for name in self.sparse_feature_names:
            # [B, (F+1), D]
            deep_out.append(self.deep_sparse[name](sparse_features[name]))
        deep_out = torch.cat(deep_out, dim=1) # [B, (F+1) * D]
        deep_out = self.mlp(deep_out) # [B, 1]

        # Final
        logits = self.bias + wide_out + deep_out
        return logits
