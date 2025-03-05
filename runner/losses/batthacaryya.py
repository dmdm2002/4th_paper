import torch
import torch.nn as nn
from einops import rearrange


class BhattacharyyaLoss(nn.Module):
    def __init__(self, eps=1e-12, use_softmax=True):
        """
        Bhattacharyya Distance Loss for Feature Matching
        :param eps: log 연산의 안정성을 위한 작은 값
        :param use_softmax: True이면 Softmax 적용하여 확률 분포로 변환
        """
        super().__init__()
        self.eps = eps  # 안정성을 위한 작은 값 추가
        self.use_softmax = use_softmax

    def forward(self, f_s, f_t):
        """
        :param f_s: Student Feature Map (batch, channel, height, width) or (batch, channel, feature_dim)
        :param f_t: Teacher Feature Map (batch, channel, height, width) or (batch, channel, feature_dim)
        :return: Bhattacharyya Distance Loss
        """
        # Feature Map을 (batch, channel, feature_dim) 형태로 변환
        if f_s.dim() == 4:
            f_s = rearrange(f_s, 'b c h w -> b c (h w)')  # (batch, channel, feature_dim)
            f_t = rearrange(f_t, 'b c h w -> b c (h w)')  # (batch, channel, feature_dim)

        # Feature를 확률 분포로 변환 (Softmax 적용)
        if self.use_softmax:
            f_s = torch.nn.functional.softmax(f_s, dim=-1)
            f_t = torch.nn.functional.softmax(f_t, dim=-1)

        # Bhattacharyya Coefficient 계산 (음수 방지)
        bc = torch.sum(torch.sqrt(torch.abs(f_s * f_t) + self.eps), dim=-1)  # (batch, channel)

        # 안정성을 위해 log에 작은 값 추가 (NaN 방지)
        loss = -torch.log(torch.clamp(bc, min=self.eps))  # (batch, channel)

        # 최종 Loss는 배치 & 채널 평균
        return loss.mean()

