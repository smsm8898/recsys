import torch

class Wide(torch.nn.Module):
    # include cross-features
    model_name = "wide"
    def __init__(
        self,
        num_sparse_features: dict[str, int],
        dense_features: list[str],
        cross_features: list[list[str]],
    ):
        super().__init__()
        self.num_sparse_features = num_sparse_features
        self.sparse_feature_names = list(num_sparse_features.keys()) # F
        self.dense_feature_names = dense_features
        self.num_dense_features = len(dense_features)


        # cross-features
        self.cross_features = cross_features
        for cross_feature in cross_features:
            cross_feature_name = f"cross_{cross_feature[0]}_{cross_feature[1]}"
            self.num_sparse_features[cross_feature_name] = self.num_sparse_features[cross_feature[0]] * self.num_sparse_features[cross_feature[1]]

        # one-hot + dense_features -> Embeddings
        self.sparse_arch = torch.nn.ModuleDict({
            name: torch.nn.Embedding(vocab_size, 1)
            for name, vocab_size in self.num_sparse_features.items() 
        })
        self.dense_arch = torch.nn.Linear(self.num_dense_features, 1, bias=False)
        

    def forward(self, features: dict[str, torch.Tensor]) -> torch.FloatTensor:
        dense_features = torch.stack([features[feature_name] for feature_name in self.dense_feature_names], dim=1)
        wide_out = self.dense_arch(dense_features) # [B, 1]
        for feature_name in self.sparse_feature_names:
            wide_out += self.sparse_arch[feature_name](features[feature_name]) # [B, 1]

        for cross_feature in self.cross_features:
            cross_feature_name = f"cross_{cross_feature[0]}_{cross_feature[1]}"
            cross_sparse_features = features[cross_feature[0]] * self.num_sparse_features[cross_feature[1]] + features[cross_feature[1]]
            wide_out += self.sparse_arch[cross_feature_name](cross_sparse_features)

        return wide_out
    
class Deep(torch.nn.Module): 
    # not include cross-features
    model_name = "deep"
    def __init__(
        self,
        num_sparse_features: dict[str, int],
        dense_features: list[str],
        latent_dim: int, 
        hidden_layers: list[int],
    ):
        super().__init__()
        self.num_sparse_features = num_sparse_features
        self.sparse_feature_names = list(num_sparse_features.keys())
        self.F = len(self.sparse_feature_names)
        self.dense_feature_names = dense_features
        self.num_dense_features = len(dense_features)
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        
        
        # Deep Embeddings
        self.sparse_arch = torch.nn.ModuleDict({
            name: torch.nn.Embedding(vocab_size, latent_dim)
            for name, vocab_size in num_sparse_features.items() 
        })
        self.dense_arch = torch.nn.Linear(self.num_dense_features, latent_dim, bias=False)

        # MLP
        mlp = []
        in_features = (self.F + 1) * latent_dim
        for out_features in hidden_layers:
            mlp.append(torch.nn.Linear(in_features, out_features, bias=False))
            mlp.append(torch.nn.ReLU())
            in_features = out_features
        if hidden_layers[-1] != 1:
            mlp.append(torch.nn.Linear(in_features, 1, bias=False))
        self.mlp = torch.nn.Sequential(*mlp)
    

    def forward(self, features: dict[str, torch.Tensor]) -> torch.FloatTensor:
        dense_features = torch.stack([features[feature_name] for feature_name in self.dense_feature_names], dim=1)
        deep_out = [self.dense_arch(dense_features)]
        for feature_name in self.sparse_feature_names:
            # [B, (F+1), D]
            deep_out.append(self.sparse_arch[feature_name](features[feature_name]))
        deep_out = torch.cat(deep_out, dim=1) # [B, (F+1) * D]
        deep_out = self.mlp(deep_out) # [B, 1]
        return deep_out
    
    
class WideAndDeep(torch.nn.Module):
    model_name = "wd"
    def __init__(
        self,
        num_sparse_features: dict[str, int],
        dense_features: list[str], 
        latent_dim: int, 
        hidden_layers: list[int],
        cross_features:list[list[str]]
    ):
        super().__init__()
        self.num_sparse_features = num_sparse_features
        self.dense_features = dense_features
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.cross_features = cross_features

        # Models
        self.bias = torch.nn.Parameter(torch.zeros([1, 1]))

        self.deep = Deep(
            num_sparse_features,
            dense_features,
            latent_dim,
            hidden_layers,
        )

        self.wide = Wide(
            num_sparse_features,
            dense_features,
            cross_features,
        )
        

    def forward(self, features: dict[str, torch.Tensor]) -> torch.FloatTensor:
        wide_out = self.wide(features)
        deep_out = self.deep(features)
        logits = self.bias + deep_out + wide_out
        return logits
        