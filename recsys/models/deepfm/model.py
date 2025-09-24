import torch

class DeepFM(torch.nn.Module):
    """
    DeepFM은 Factorization Machine(FM)과 딥러닝(MLP)의 장점을 모두 가집니다.
    피처의 1차, 2차, 그리고 고차 상호작용까지 학습합니다.
    Embedding Table을 FM과 Deep에서 모두 공유합니다.
    """
    def __init__(self, num_features: int, latent_dim: int, hidden_layers: list[int]):
        super().__init__()

        # FM Component
        self.W0 = torch.nn.Parameter(torch.zeros(1))
        self.W = torch.nn.Embedding(num_features, 1)
        self.V = torch.nn.Embedding(num_features, latent_dim)

        # Deep: MLP(Embedding은 공유)
        mlp = []
        in_featues = len(hidden_layers) * latent_dim
        for out_featues in hidden_layers:
            mlp.append(torch.nn.Linear(in_featues, out_featues))
            mlp.append(torch.nn.ReLU())
            in_featues = out_featues
        self.mlp = torch.nn.ModuleList(mlp)

        # Prediction Layer
        final_dim = 1 + latent_dim + hidden_layers[-1]
        self.classifier = torch.nn.Linear(final_dim, 1)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        # FM 
        fm_first_term = self.W(x).sum(dim=1).unsqueeze(1)
        
        v = self.V(x) # [batch_size, num_fields, latent_dim]
        sum_of_square = v.sum(dim=1).pow(2) # [batch_size, latent_dim]
        squares_of_sum = v.pow(2).sum(dim=1) # [batch_size, latent_dim]
        fm_second_term = 0.5 * (sum_of_square - squares_of_sum).sum(dim=1, keepdim=True)
        fm_out = self.W0 + fm_first_term + fm_second_term

        # Deep
        deep_out = torch.flatten(v, 1) # [batch_size, num_fields * latent_dim]
        for m in self.mlp:
            deep_out = m(deep_out)
        
        # Final
        concat = torch.cat([fm_out, deep_out], dim=-1)
        logits = self.classifier(concat)
        return logits
