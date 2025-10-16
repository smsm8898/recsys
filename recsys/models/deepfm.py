import torch
# 참고 https://github.com/meta-pytorch/torchrec/blob/main/torchrec/modules/deepfm.py

class FM(torch.nn.Module):
    model_name = "fm"
    def __init__(self):
        super().__init__()
        
        
    def forward(self, embeddings: torch.FloatTensor) -> torch.FloatTensor:
        sum_of_square = embeddings.sum(dim=1, keepdim=True).pow(2) # [B, 1]
        squares_of_sum = embeddings.pow(2).sum(dim=1, keepdim=True) # [B, 1]
        fm_out = 0.5 * (sum_of_square - squares_of_sum).sum(dim=1, keepdim=True) # [B, 1]
        return fm_out
        
    
class Deep(torch.nn.Module):
    model_name = "deep"
    def __init__(
        self,
        F: int,
        latent_dim: int, 
        hidden_layers: list[int],
    ):
        super().__init__()
        self.F = F
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        
        # MLP
        mlp = []
        in_features = (F + 1) * latent_dim
        for out_features in hidden_layers:
            mlp.append(torch.nn.Linear(in_features, out_features, bias=False))
            mlp.append(torch.nn.ReLU())
            in_features = out_features
        self.mlp = torch.nn.Sequential(*mlp)

    def forward(self, embeddings: torch.FloatTensor) -> torch.FloatTensor:
        deep_out = self.mlp(embeddings)
        return deep_out
        

class DeepFM(torch.nn.Module):
    model_name = "deepfm"
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

        
        # Input Embedding(Common)
        self.sparse_arch = torch.nn.ModuleDict({
            name: torch.nn.Embedding(vocab_size, latent_dim)
            for name, vocab_size in num_sparse_features.items() 
        })
        self.dense_arch = torch.nn.Linear(self.num_dense_features, latent_dim, bias=False)

        # DeepFM
        self.deep = Deep(
            self.F,
            latent_dim,
            hidden_layers,
        )
        self.fm = FM()

        # Classifier(dense_feature + fm_out + deep_out)
        ### dense_features는 1st order interaction(addition unit)
        in_features = latent_dim + hidden_layers[-1] + 1
        self.classifier = torch.nn.Linear(in_features, 1)
        

    def forward(self, features: dict[str, torch.Tensor]) -> torch.FloatTensor:
        dense_features = torch.stack([features[feature_name] for feature_name in self.dense_feature_names], dim=1)
        dense_features = self.dense_arch(dense_features)
        
        # common input: embeddings
        embeddings = [dense_features] # [B, D]
        for feature_name in self.sparse_feature_names:
            embeddings.append(self.sparse_arch[feature_name](features[feature_name]))
        embeddings = torch.cat(embeddings, dim=1) # [B, (F+1) * D]
        
        # FM
        fm_out = self.fm(embeddings) # [B, 1]
        
        # Deep
        deep_out = self.deep(embeddings) # [B, H]

        # Classifier
        interactions = torch.cat([dense_features, fm_out, deep_out], dim=1) # [B, D+H+1]
        logits = self.classifier(interactions)
        return logits 