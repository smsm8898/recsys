import torch

class WideAndDeep(torch.nn.Module):
    def __init__(
        self,
        num_sparse_fields: int,
        num_sparse_features: int,
        num_dense_features: int,
        latent_dim: int,
        hidden_layers: list[int]
    ):
        super().__init__()
        self.num_sparse_fields = num_sparse_fields
        self.num_sparse_features = num_sparse_features
        self.num_dense_features = num_dense_features
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers

        # Wide Component: Linear model
        self.wide_sparse = torch.nn.Embedding(num_sparse_features, 1)
        self.wide_dense = torch.nn.Linear(num_dense_features, 1)

        # Deep Component: Embedding + MLP
        self.deep_embedding = torch.nn.Embedding(num_sparse_features, latent_dim)

        mlp = []
        in_features = num_sparse_fields * latent_dim + num_dense_features
        for out_features in hidden_layers:
            mlp.append(torch.nn.Linear(in_features, out_features))
            mlp.append(torch.nn.ReLU())
            in_features = out_features
        self.mlp = torch.nn.ModuleList(mlp)

        # Prediction Layer
        final_dim = 1 + hidden_layers[-1]
        self.classifier = torch.nn.Linear(final_dim, 1)

    def forward(self, sparse_features: torch.LongTensor, dense_features: torch.FloatTensor) -> torch.FloatTensor:
        # Wide
        wide_sparse_out = self.wide_sparse(sparse_features).sum(dim=1) # [batch_size, 1]
        wide_dense_out = self.wide_dense(dense_features) # [batch_size, 1]
        wide_out = wide_sparse_out + wide_dense_out # [batch_size, 1]

        # Deep
        deep_sparse_embeddings = self.deep_embedding(sparse_features) # [batch_size, num_sparse_fields, latent_dim]
        deep_sparse_embeddings = torch.flatten(deep_sparse_embeddings, 1) # [batch_size, num_sparse_fields * latent_dim]
        deep_out = torch.cat([deep_sparse_embeddings, dense_features], dim=1) # [batch_size, num_sparse_fields * latent_dim + num_dense_features]
        for m in self.mlp:
            deep_out = m(deep_out)
        
        # Final
        concat = torch.cat([wide_out, deep_out], dim=-1)
        logits = self.classifier(concat)
        return logits
