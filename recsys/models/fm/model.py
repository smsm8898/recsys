import torch

class FM(torch.nn.Module):
    """
    Factorization Machine(FM)은 희소 데이터에서 피처 간의 상호작용을 모델링하는 데 특화된 모델입니다.
    특히, 아직 한 번도 함께 발생하지 않은 피처 조합(예: 새로운 사용자와 새로운 영화)에 대한 예측도 가능하게 해준다는 장점이 있습니다.
    """
    def __init__(self, num_features: int, latent_dim: int):
        super().__init__()
        self.num_features = num_features
        self.latent_dim = latent_dim

        self.W0 = torch.nn.Parameter(torch.zeros(1)) # global bias
        self.W = torch.nn.Embedding(num_features, 1) 
        self.V = torch.nn.Embedding(num_features, latent_dim)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        first_term = self.W(x).sum(dim=1)
        
        v = self.V(x) # [batch_size, num_fields, latent_dim]
        sum_of_square = v.sum(dim=1).pow(2) # [batch_size, latent_dim]
        squares_of_sum = v.pow(2).sum(dim=1) # [batch_size, latent_dim]
        second_term = 0.5 * (sum_of_square - squares_of_sum).sum(dim=1, keepdim=True)

        logits = self.W0 + first_term + second_term
        return logits