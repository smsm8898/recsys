import torch

class DeepCrossNetwork(torch.nn.Module):
    def __init__(
        self,
        num_sparse_fields: int,
        num_sparse_features: int,
        num_dense_features: int,
        latent_dim: int,
        num_cross_layers: int,
        hidden_layers: list[int],
        *args,
        **kwargs,
    ):
        super().__init__()
        self.num_sparse_fields = num_sparse_fields
        self.num_sparse_features = num_sparse_features
        self.num_dense_features = num_dense_features
        self.latent_dim = latent_dim
        self.num_cross_layers = num_cross_layers
        self.hidden_layers = hidden_layers

        # Input Embedding Layer(공유)
        self.W_sparse = torch.nn.Embedding(num_sparse_features, latent_dim)
        self.W_dense = torch.nn.Linear(num_dense_features, latent_dim)

        # Cross Network
        in_features = (num_sparse_fields + 1) * latent_dim
        self.cross_kernels = torch.nn.ParameterList([
            torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(in_features, in_features)))
            for _ in range(num_cross_layers)
        ])
        self.cross_bias = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(in_features, 1))
            for _ in range(num_cross_layers)
        ])

        # Deep Network: MLP(Embedding은 공유)
        mlp = []
        mlp_in_features = in_features
        for mlp_out_features in hidden_layers:
            mlp.append(torch.nn.Linear(mlp_in_features, mlp_out_features))
            mlp.append(torch.nn.ReLU())
            mlp_in_features = mlp_out_features
        self.mlp = torch.nn.ModuleList(mlp)

        # Prediction Layer
        final_dim = in_features + hidden_layers[-1]
        self.classifier = torch.nn.Linear(final_dim, 1)

    def forward(self, sparse_features: torch.LongTensor, dense_features: torch.FloatTensor) -> torch.FloatTensor:
        # 2-1. Embedding
        x0_sparse = self.W_sparse(sparse_features) # [batch_size, num_sparse_fields, latent_dim]
        x0_dense = self.W_dense(dense_features) # [batch_size, latent_dim]
        x0 = torch.cat([torch.flatten(x0_sparse, 1), x0_dense], dim=1) # [batch_size, (num_sparse_fields + 1) * latent_dim]

        # 2. Cross Network
        x_0 = x0.unsqueeze(2)  # [batch_size, (num_sparse_fields + 1) * latent_dim, 1]
        x_l = x_0
        for layer in range(self.num_cross_layers):
            xl_w = torch.matmul(self.cross_kernels[layer], x_l)  # [batch_size, (num_sparse_fields + 1) * latent_dim, 1]
            x_l = x_0 * (xl_w + self.cross_bias[layer]) + x_l     # [batch_size, (num_sparse_fields + 1) * latent_dim, 1]
        cross_out = torch.squeeze(x_l, dim=2)                # [batch_size, (num_sparse_fields + 1) * latent_dim]

        # 2-3. Deep Network
        deep_out = x0
        for m in self.mlp:
            deep_out = m(deep_out)

        # 2-4. Combination Layer
        concat = torch.cat([cross_out, deep_out], dim=-1)
        logits = self.classifier(concat)
        return logits