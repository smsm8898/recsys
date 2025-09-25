import torch

class NeuralCollaborativeFiltering(torch.nn.Module):
    def __init__(
        self,
        num_sparse_features: int,
        latent_dim: int,
        hidden_layers: list[int]
    ):
        super().__init__()
        # Embedding
        self.W_gmf = torch.nn.Embedding(num_sparse_features, latent_dim)
        self.W_mlp = torch.nn.Embedding(num_sparse_features, latent_dim)

        # MLP Component
        mlp = []
        in_features = 2 * latent_dim
        for out_features in hidden_layers:
            mlp.append(torch.nn.Linear(in_features, out_features))
            mlp.append(torch.nn.ReLU())
            in_features = out_features
        self.mlp = torch.nn.ModuleList(mlp)

        # Prediction Layer
        final_dim = latent_dim + hidden_layers[-1]
        self.classifier = torch.nn.Linear(final_dim, 1)

    def forward(self, sparse_features: torch.LongTensor) -> torch.FloatTensor:

        # GMF
        gmf_embeddings = self.W_gmf(sparse_features) # [batch_size, 2, latent_dim]
        gmf_out = gmf_embeddings[:, 0, :] * gmf_embeddings[:, 1, :] # [batch_size, latent_dim]

        # MLP
        mlp_embeddings = self.W_mlp(sparse_features) # [batch_size, 2, latent_dim]
        # mlp_out = torch.cat([mlp_user_emb, mlp_item_emb], dim=-1)
        mlp_out = torch.flatten(mlp_embeddings, 1) # [batch_size, 2 * latent_dim]
        for m in self.mlp:
            mlp_out = m(mlp_out)

        concat = torch.cat([gmf_out, mlp_out], dim=-1)
        logits = self.classifier(concat)
        return logits
