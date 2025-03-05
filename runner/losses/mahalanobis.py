import torch
import torch.nn as nn
from einops import rearrange


class MahalanobisLoss(nn.Module):
    def __init__(self, eps=1e-6):
        """
        Mahalanobis Distance Loss (Batch-wise)
        :param eps: 공분산 행렬의 안정성을 위한 작은 값
        """
        super().__init__()
        self.eps = eps

    def forward(self, f_s, f_t):
        """
        :param f_s: Student feature map (batch, channel, height, width)
        :param f_t: Teacher feature map (batch, channel, height, width)
        :return: Mahalanobis distance loss
        """
        b, c, h, w = f_s.shape  # (batch, channel, height, width)

        # ✅ Feature Map을 (batch, channel, feature_dim) 형태로 변환
        f_s = rearrange(f_s, 'b c h w -> b c (h w)')  # (batch, channel, feature_dim)
        f_t = rearrange(f_t, 'b c h w -> b c (h w)')  # (batch, channel, feature_dim)

        # ✅ Batch-wise로 개별 공분산 행렬 계산
        mahalanobis_distances = []
        for i in range(b):  # 배치 내 개별 샘플마다 독립적인 공분산 행렬 계산
            mu_t = torch.mean(f_t[i], dim=1, keepdim=True)  # (channel, 1)
            diff_t = f_t[i] - mu_t  # (channel, feature_dim)

            # ✅ 공분산 행렬 계산 (Batch-wise)
            cov_t = (diff_t @ diff_t.T) / (diff_t.shape[1] - 1)  # (channel, channel)

            # ✅ 안정성을 위해 작은 값(eps) 추가 후 역행렬 계산
            cov_t_inv = torch.linalg.pinv(
                cov_t + self.eps * torch.eye(cov_t.shape[0], device=f_t.device)
            )

            # ✅ Mahalanobis 거리 계산
            diff_s = f_s[i] - mu_t  # (channel, feature_dim)
            mahalanobis_dist = torch.sqrt(torch.sum((diff_s.T @ cov_t_inv) * diff_s.T, dim=1) + self.eps)  # (feature_dim,)
            mahalanobis_distances.append(torch.mean(mahalanobis_dist))  # 평균 거리 저장

        # ✅ 배치 평균 Loss 반환
        loss = torch.mean(torch.stack(mahalanobis_distances))

        return loss

# class MahalanobisLoss(nn.Module):
#     def __init__(self, eps=1e-6):
#         super().__init__()
#         self.eps = eps # 공분산 행렬의 안정성을 위한 값
#
#     def forward(self, f_s, f_t):
#         """
#         :param f_s: Student feature map (batch, channel, height, width)
#         :param f_t: Teacher feature map (batch, channel, height, width)
#         :return: Mahalanobis distance
#         """
#         # (b, c, h, w) = f_s.shape
#
#         f_s = rearrange(f_s, 'b c h w -> b c (h w)')
#         f_t = rearrange(f_t, 'b c h w -> b c (h w)')
#
#         # Feature Map Average Calculate (Teacher)
#         mu_t = torch.mean(f_t, dim=0, keepdim=True)
#
#         # 공분산 행렬 계산 (Teacher)
#         diff_t = f_t - mu_t
#         cov_t = (diff_t.T @ diff_t) / (f_t.shape[0] - 1)
#
#         # 공분산 행렬의 역행렬 계산 (안정성을 위한 eps 값 추가, SVD로 역행렬 구함)
#         cov_t_inv = torch.linalg.pinv(
#             cov_t + self.eps * torch.eye(cov_t.shape[0], device=f_t.device)
#         )
#
#         # Mahalanobis 거리 계산
#         diff_s = f_s - mu_t  # (16*112*112, 64)
#         mahalanobis_dist = torch.sqrt(torch.sum((diff_s @ cov_t_inv) * diff_s, dim=1))  # (16*112*112, )
#
#         # # 최종 Loss (평균 거리)
#         loss = torch.mean(mahalanobis_dist)
#
#         return loss

