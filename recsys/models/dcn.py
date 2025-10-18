import torch


class DeepCrossNetwork(torch.nn.Module):
    model_name = "dcn"
    def __init__(
        self,
        num_sparse_features: dict[str, int],
        dense_features: list[str], 
        latent_dim: int, 
        hidden_layers: list[int],
        num_cross_layers: int,
    ):
        super().__init__()
        self.num_sparse_features = num_sparse_features
        self.sparse_feature_names = list(num_sparse_features.keys())
        self.F = len(self.sparse_feature_names)
        self.dense_feature_names = dense_features
        self.num_dense_features = len(dense_features)
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.num_cross_layers = num_cross_layers

        # Embedding and stacking layer
        self.sparse_arch = torch.nn.ModuleDict({
            name: torch.nn.Embedding(vocab_size, latent_dim)
            for name, vocab_size in num_sparse_features.items() 
        })
        self.dense_arch = torch.nn.Linear(self.num_dense_features, latent_dim, bias=False)

        # Cross Network
        in_features = (self.F + 1) * latent_dim
        self.cross_network = torch.nn.ModuleDict({
            "w" : torch.nn.ParameterList([
                torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(in_features, in_features)))
                for _ in range(num_cross_layers)
            ]),
            "b": torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(in_features, 1))
            for _ in range(num_cross_layers)
            ])
        })

        # Deep Network
        mlp = []
        in_features = (self.F + 1) * latent_dim
        for out_features in hidden_layers:
            mlp.append(torch.nn.Linear(in_features, out_features, bias=False))
            mlp.append(torch.nn.ReLU())
            in_features = out_features
        self.mlp = torch.nn.Sequential(*mlp)

        # Combination output layer
        final_dim = (self.F + 1) * latent_dim + hidden_layers[-1]
        self.classifier = torch.nn.Linear(final_dim, 1, bias=True)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.FloatTensor:
        # 1. Embedding and stacking layer
        x0 = torch.stack([features[feature_name] for feature_name in self.dense_feature_names], dim=1)
        x0 = [self.dense_arch(x0)]
        for feature_name in self.sparse_feature_names:
            x0.append(self.sparse_arch[feature_name](features[feature_name]))
        x0 = torch.cat(x0, dim=1) # [B, (F+1) * D]
        

        # 2. Cross Network
        x_0 = x0.unsqueeze(2)  # [B, (F + 1) * D, 1]
        x_l = x_0
        for layer in range(self.num_cross_layers):
            xl_w = torch.matmul(self.cross_network["w"][layer], x_l)  # [B, (F + 1) * D, 1]
            x_l = x_0 * (xl_w + self.cross_network["b"][layer]) + x_l
        cross_out = torch.squeeze(x_l, dim=2)

        # 2-3. Deep Network
        deep_out = x0
        for m in self.mlp:
            deep_out = m(deep_out)

        # 2-4. Combination output layer
        concat = torch.cat([cross_out, deep_out], dim=-1)
        logits = self.classifier(concat)
        return logits