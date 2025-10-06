import torch

class DeepFM(torch.nn.Module):
    model_name = "deepfm"
    def __init__(
        self,
        num_sparse_fields: int,
        num_sparse_features: int,
        num_dense_features: int,
        latent_dim: int,
        hidden_layers: list[int],
        *args,
        **kwargs,
    ):
        super().__init__()
        self.num_sparse_fields = num_sparse_fields
        self.num_sparse_features = num_sparse_features
        self.num_dense_features = num_dense_features
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers

        # FM Component
        self.W0 = torch.nn.Parameter(torch.zeros(1))
        self.W_sparse = torch.nn.Embedding(num_sparse_features, 1)
        self.W_dense = torch.nn.Linear(num_dense_features, 1)
        self.V_sparse = torch.nn.Embedding(num_sparse_features, latent_dim)
        self.V_dense = torch.nn.Linear(num_dense_features, latent_dim, bias=False)

        # Deep: MLP(Embedding은 공유)
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
        # FM 
        fm_first_sparse_term = self.W_sparse(sparse_features).sum(dim=1) # [batch_size, 1]
        fm_first_dense_term = self.W_dense(dense_features) # [batch_size, 1]
        fm_first_term = fm_first_sparse_term + fm_first_dense_term # [batch_size, 1]
        
        v_sparse = self.V_sparse(sparse_features) # [batch_size, num_sparse_fields, latent_dim]
        v_dense = self.V_dense(dense_features).unsqueeze(1) # [batch_size, 1, latent_dim]
        v = torch.cat([v_sparse, v_dense], dim=1) # [batch_size, num_sparse_fields+1, latent_dim]
        
        sum_of_square = v.sum(dim=1).pow(2) # [batch_size, latent_dim]
        squares_of_sum = v.pow(2).sum(dim=1) # [batch_size, latent_dim]
        fm_second_term = 0.5 * (sum_of_square - squares_of_sum).sum(dim=1, keepdim=True) # [batch_size, 1]
        fm_out = self.W0 + fm_first_term + fm_second_term 

        # Deep
        deep_out = torch.flatten(v_sparse, 1) # [batch_size, num_fields * latent_dim]
        deep_out = torch.cat([deep_out, dense_features], dim=1) # [batch_size, num_fields * latent_dim + num_dense_features]
        for m in self.mlp:
            deep_out = m(deep_out)
        
        # Final
        concat = torch.cat([fm_out, deep_out], dim=-1)
        logits = self.classifier(concat)
        return logits
