import torch

class DCN(torch.nn.Module):
    """
    DCN은 피처 간의 상호작용을 자동으로 학습하는 'Cross Network'를 포함합니다.
    Cross Network는 피처의 고차 상호작용을 명시적으로 모델링합니다.
    Embedding Table을 FM과 Deep에서 모두 공유합니다.
    """
    def __init__(self, num_features: int, latent_dim: int, hidden_layers: list[int], num_cross_layers: int):
        super().__init__()

        # Embedding Layer(공유)
        self.W = torch.nn.Embedding(num_features, latent_dim)

        # Cross Network
        cross_layers = []
        in_featues = len(hidden_layers) * latent_dim
        for _ in num_cross_layers:
            cross_layers.append(torch.nn.Linear(in_featues, in_featues))
        self.cross_layers = torch.nn.ModuleList(cross_layers)

        # Deep: MLP(Embedding은 공유)
        mlp = []
        for out_featues in hidden_layers:
            mlp.append(torch.nn.Linear(in_featues, out_featues))
            mlp.append(torch.nn.ReLU())
            in_featues = out_featues
        self.mlp = torch.nn.ModuleList(mlp)

        # Prediction Layer
        final_dim = len(hidden_layers) * latent_dim + hidden_layers[-1]
        self.classifier = torch.nn.Linear(final_dim, 1)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        # 2-1. Embedding
        x0 = self.W(x)
        x0 = torch.flatten(x, 1) # [batch_size, num_fields * latent_dim]

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