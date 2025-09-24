import torch

class WD(torch.nn.Module):
    """
    Wide & Deep 모델은 선형 모델의 '기억(memorization)' 능력과 
    딥러닝 모델의 '일반화(generalization)' 능력 모두를 활용합니다.
    모든 피처를 Wide 컴포넌트에 직접 입력하고, 동시에 이 피처들의 임베딩을 Deep 컴포넌트로 전달합니다.
    """
    def __init__(self, num_features: int, wide_dim: int, deep_dim: int, hidden_layers: list[int]):
        super().__init__()

        # Wide Component: Linear model
        self.wide_component = torch.nn.Embedding(num_features, wide_dim)

        # Deep Component: Embedding + MLP
        self.deep_embedding = torch.nn.Embedding(num_features, deep_dim)

        mlp = []
        in_featues = len(hidden_layers) * deep_dim
        for out_featues in hidden_layers:
            mlp.append(torch.nn.Linear(in_featues, out_featues))
            mlp.append(torch.nn.ReLU())
            in_featues = out_featues
        self.mlp = torch.nn.ModuleList(mlp)

        # Prediction Layer
        final_dim = wide_dim + hidden_layers[-1]
        self.classifier = torch.nn.Linear(final_dim, 1)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        # Wide
        wide_out = self.wide_component(x).sum(dim=1).unsqueeze(1)

        # Deep
        deep_out = self.deep_embedding(x) # [batch_size, num_fields, latent_dim]
        deep_out = torch.flatten(deep_out, 1) # [batch_size, num_fields * latent_dim]
        for m in self.mlp:
            deep_out = m(deep_out)
        
        # Final
        concat = torch.cat([wide_out, deep_out], dim=-1)
        logits = self.classifier(concat)
        return logits
