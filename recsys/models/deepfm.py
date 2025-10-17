import torch


class DeepFM(torch.nn.Module):
    model_name = "deepfm"
    def __init__(
        self,
        num_sparse_features: dict[str, int],
        dense_features: list[str], 
        latent_dim: int, 
        hidden_layers: list[int],
        dropout: float,
    ):
        super().__init__()
        self.num_sparse_features = num_sparse_features
        self.sparse_feature_names = list(num_sparse_features.keys())
        self.F = len(self.sparse_feature_names)
        self.dense_feature_names = dense_features
        self.num_dense_features = len(dense_features)
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout

        # global bias
        self.bias = torch.nn.Parameter(torch.zeros(1, 1))

        # common-input for (fm - 2nd: cross) and (deep)
        self.sparse_arch = torch.nn.ModuleDict({
            name: torch.nn.Embedding(vocab_size, latent_dim)
            for name, vocab_size in num_sparse_features.items() 
        })
        self.dense_arch = torch.nn.Linear(self.num_dense_features, latent_dim, bias=False)

        # addition unit(fm - 1st: linear)
        self.linear_sparse = torch.nn.ModuleDict({
            name: torch.nn.Embedding(num_sparse_feature, 1)
            for name, num_sparse_feature in num_sparse_features.items()
        })
        self.linear_dense = torch.nn.Linear(self.num_dense_features, 1)

        # deep
        mlp = []
        in_features = (self.F + 1) * latent_dim
        for out_features in hidden_layers:
            mlp.append(torch.nn.Linear(in_features, out_features, bias=False))
            mlp.append(torch.nn.ReLU())
            mlp.append(torch.nn.Dropout(dropout)) # experiment에 존재
            in_features = out_features
        if hidden_layers[-1] != -1:
            mlp.append(torch.nn.Linear(in_features, 1, bias=False))
        self.mlp = torch.nn.Sequential(*mlp)

        
    def forward(self, features: dict[str, torch.Tensor]) -> torch.FloatTensor:
        # Dense Embeddings
        dense_features = torch.stack([features[feature_name] for feature_name in self.dense_feature_names], dim=1)
        embeddings = [self.dense_arch(dense_features)]
        for feature_name in self.sparse_feature_names:
            embeddings.append(self.sparse_arch[feature_name](features[feature_name]))
        embeddings = torch.stack(embeddings, dim=1) # [B, (F+1), D]
        
        # FM Layer
        # - 1st: linear
        first_term = [self.linear_dense(dense_features)]
        for feature_name in self.sparse_feature_names:
            # [B, (F+1), 1]
            first_term.append(self.linear_sparse[feature_name](features[feature_name])) 
        first_term = torch.cat(first_term, dim=1).sum(dim=1, keepdim=True) # [B, 1]
        
        # - 2nd: cross
        sum_of_square = embeddings.sum(dim=1).pow(2) # [B, D]
        squares_of_sum = embeddings.pow(2).sum(dim=1) # [B, D]
        second_term = 0.5 * (sum_of_square - squares_of_sum).sum(dim=1, keepdim=True) # [B, 1]

        # Hidden Layer
        deep_out = self.mlp(torch.flatten(embeddings, 1)) # [B, 1]

        # Output Units
        logits = self.bias + first_term + second_term + deep_out
        return logits