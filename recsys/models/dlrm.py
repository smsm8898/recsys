import torch


class DeepLearningRecommenderModel(torch.nn.Module):
    model_name = "dlrm"
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


        # Bottom Arch - simple version(fixed mlp)
        self.sparse_arch = torch.nn.ModuleDict({
            name: torch.nn.Embedding(vocab_size, latent_dim)
            for name, vocab_size in num_sparse_features.items() 
        })
        self.dense_arch = torch.nn.Sequential(
            torch.nn.Linear(self.num_dense_features, latent_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(latent_dim, latent_dim),
            torch.nn.ReLU(),
        )

        # Interaction Arch - pairwise dot product
        ### F+1개의 feature 중 2개를 선택하는 indices
        self.register_buffer(
            "triu_indices",
            torch.triu_indices(self.F + 1, self.F + 1, offset=1),
            persistent=False
        )
        num_pairs = int(self.F * (self.F+1) / 2) # sparse + dense -> (F+1) choose 2

        # Over Arch
        over_arch = []
        in_features = (self.F + 1) * latent_dim + num_pairs
        for out_features in hidden_layers:
            over_arch.append(torch.nn.Linear(in_features, out_features))
            over_arch.append(torch.nn.ReLU())
            in_features = out_features
        if hidden_layers[-1] != -1:
            over_arch.append(torch.nn.Linear(in_features, 1))
        self.over_arch = torch.nn.Sequential(*over_arch)

        
    def forward(self, features: dict[str, torch.Tensor]) -> torch.FloatTensor:
        # Bottom
        dense_features = torch.stack([features[feature_name] for feature_name in self.dense_feature_names], dim=1)
        embeddings = [self.dense_arch(dense_features)]
        for feature_name in self.sparse_feature_names:
            embeddings.append(self.sparse_arch[feature_name](features[feature_name]))
        embeddings = torch.stack(embeddings, dim=1) # [B, (F+1), D]
        
        # Interaction(pairwise dot-product)
        interactions = torch.bmm(embeddings, embeddings.transpose(1, 2)) # [B, F+1, F+1]
        interactions = interactions[:, self.triu_indices[0], self.triu_indices[1]] # [B, F * (F+1) / 2]
        embeddings_interactions = torch.cat(
            [torch.flatten(embeddings, 1), interactions],
        dim=1)

        # Over Arch(MLP)
        logits = self.over_arch(embeddings_interactions)
        return logits