import torch

class Wide(torch.nn.Module):
    model_name = "wide"
    def __init__(
        self,
        num_sparse_features: dict[str, int],
        num_dense_features: int
    ):
        super().__init__()
        # include cross-features
        self.num_sparse_features = num_sparse_features
        self.sparse_feature_names = list(num_sparse_features.keys())

        # one-hot + dense_features
        in_features = sum(num_sparse_features.values()) + num_dense_features
        self.linear_arch = torch.nn.Linear(in_features, 1, bias=False)

    def forward(self, sparse_features: dict[str, torch.LongTensor], dense_features: torch.FloatTensor) -> torch.FloatTensor:
        wide_out = [dense_features]
        for name in self.sparse_feature_names:
            wide_out.append(
                torch.nn.functional.one_hot(sparse_features[name], self.num_sparse_features[name]).float()
            )
        wide_out = torch.cat(wide_out, dim=1)
        wide_out = self.linear_arch(wide_out)
        return wide_out
            
class Deep(torch.nn.Module):
    model_name = "deep"
    def __init__(
        self,
        num_sparse_features: dict[str, int],
        num_dense_features: int, 
        latent_dim: int, 
        hidden_layers: list[int],
    ):
        super().__init__()
        # exclude cross-features
        self.num_sparse_features = num_sparse_features
        self.sparse_feature_names = list(num_sparse_features.keys())
        self.F = len(self.sparse_feature_names)
        self.num_dense_features = num_dense_features
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        
        
        # Deep Embeddings
        self.sparse_arch = torch.nn.ModuleDict({
            name: torch.nn.Embedding(vocab_size, latent_dim)
            for name, vocab_size in num_sparse_features.items() 
        })
        self.dense_arch = torch.nn.Linear(num_dense_features, latent_dim)

        # MLP
        mlp = []
        in_features = (self.F + 1) * latent_dim
        for out_features in hidden_layers:
            mlp.append(torch.nn.Linear(in_features, out_features))
            mlp.append(torch.nn.ReLU())
            in_features = out_features
        if hidden_layers[-1] != 1:
            mlp.append(torch.nn.Linear(in_features, 1, bias=False))
        self.mlp = torch.nn.Sequential(*mlp)
    

    def forward(self, sparse_features: dict[str, torch.LongTensor], dense_features: torch.FloatTensor) -> torch.FloatTensor:
        deep_out = [self.dense_arch(dense_features)]
        for name in self.sparse_feature_names:
            # [B, (F+1), D]
            deep_out.append(self.sparse_arch[name](sparse_features[name]))
        deep_out = torch.cat(deep_out, dim=1) # [B, (F+1) * D]
        deep_out = self.mlp(deep_out) # [B, 1]
        return deep_out

class WideAndDeep(torch.nn.Module):
    model_name = "wd"
    def __init__(
        self,
        num_sparse_features: dict[str, int],
        num_dense_features: int, 
        latent_dim: int, 
        hidden_layers: list[int],
        cross_prefix: str = "cross"
    ):
        super().__init__()
        self.num_sparse_features = num_sparse_features
        self.num_dense_features = num_dense_features
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.cross_prefix = cross_prefix
    
        # Models
        self.wide = Wide(
            # include cross features
            num_sparse_features,
            num_dense_features
        )
        
        self.deep = Deep(
            # exclude cross features
            {k:v for k, v in num_sparse_features.items() if self.cross_prefix not in k},
            num_dense_features,
            latent_dim,
            hidden_layers
        )
        self.bias = torch.nn.Parameter(torch.zeros([1, 1]))

    def forward(self, sparse_features: dict[str, torch.LongTensor], dense_features: torch.FloatTensor) -> torch.FloatTensor:
        wide_out = self.wide(sparse_features, dense_features)
        deep_out = self.deep(sparse_features, dense_features)
        logits = self.bias + wide_out + deep_out
        return logits
        