import torch

class GeneralizedMatrixFactorization(torch.nn.Module):
    model_name = "gmf"
    def __init__(self, num_sparse_features: dict[str, int], latent_dim: int):
        super().__init__()
        self.num_sparse_features = num_sparse_features
        self.latent_dim = latent_dim
        self.sparse_feature_names = list(num_sparse_features.keys())

        self.sparse_arch = torch.nn.ModuleDict({
            name: torch.nn.Embedding(num_sparse_feature, latent_dim)
            for name, num_sparse_feature in num_sparse_features.items()
        })

    def forward(self, features: dict[str, torch.LongTensor]) -> torch.FloatTensor:
        emb1 = self.sparse_arch[self.sparse_feature_names[0]](features[self.sparse_feature_names[0]]) # [B, D]
        emb2 = self.sparse_arch[self.sparse_feature_names[1]](features[self.sparse_feature_names[1]]) # [B, D]

        # Element-wise Product
        out = (emb1 * emb2) # [B, D] 
        return out
    
class NeuralCollaborativeFiltering(torch.nn.Module):
    model_name = "ncf"
    def __init__(self, num_sparse_features: dict[str, int], latent_dim: int, hidden_layers: list[int]):
        super().__init__()
        self.num_sparse_features = num_sparse_features
        self.latent_dim = latent_dim
        self.sparse_feature_names = list(num_sparse_features.keys())

        self.sparse_arch = torch.nn.ModuleDict({
            name: torch.nn.Embedding(num_sparse_feature, latent_dim)
            for name, num_sparse_feature in num_sparse_features.items()
        })

        # MLP layers
        mlp = []
        input_dim = 2 * latent_dim
        for h in hidden_layers:
            mlp.append(torch.nn.Linear(input_dim, h))
            mlp.append(torch.nn.ReLU())
            input_dim = h
        self.mlp = torch.nn.Sequential(*mlp)

    def forward(self, features: dict[str, torch.LongTensor]) -> torch.FloatTensor:
        emb1 = self.sparse_arch[self.sparse_feature_names[0]](features[self.sparse_feature_names[0]]) # [B, D]
        emb2 = self.sparse_arch[self.sparse_feature_names[1]](features[self.sparse_feature_names[1]]) # [B, D]

        # Concatenation
        concat = torch.cat([emb1, emb2], dim=-1) # [B, 2*D]
        out = self.mlp(concat) # [B, H] H is last-hidden
        return out
    
class NeuMF(torch.nn.Module):
    model_name = "neumf"
    def __init__(self, num_sparse_features: dict[str, int], latent_dim: int, hidden_layers: list[int]):
        super().__init__()
        self.gmf = GeneralizedMatrixFactorization(num_sparse_features, latent_dim)
        self.ncf = NeuralCollaborativeFiltering(num_sparse_features, latent_dim, hidden_layers)


        # NeuMF Layer
        final_dim = latent_dim + hidden_layers[-1] # concat of MF and NCF outputs
        self.classifier = torch.nn.Linear(final_dim, 1)
        
        
    def forward(self, features: dict[str, torch.LongTensor]):
        gmf_out = self.gmf(features) # [B, D]
        ncf_out = self.ncf(features) # [B, H]
        
        # concatenate MF latent and NCF hidden representation
        concat = torch.cat([gmf_out, ncf_out], dim=-1) # [B, (D + H)]
        logits = self.classifier(concat)
        return logits