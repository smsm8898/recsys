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
        cross_layers = []
        in_features = (num_sparse_fields + 1) * latent_dim
        for _ in range(num_cross_layers):
            cross_layers.append(torch.nn.Linear(in_features, in_features))
        self.cross_layers = torch.nn.ModuleList(cross_layers)

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

        # 2-2. Cross Network
        cross_out = x0
        for c in self.cross_layers:
            cross_out = x0 * c(cross_out) + cross_out

        # 2-3. Deep Network
        deep_out = x0
        for m in self.mlp:
            deep_out = m(deep_out)

        # 2-4. Combination Layer
        concat = torch.cat([cross_out, deep_out], dim=-1)
        logits = self.classifier(concat)
        return logits