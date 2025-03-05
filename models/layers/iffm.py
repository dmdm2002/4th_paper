import torch
import torch.nn as nn
import torch.fft as fft


# Important Feature Frequency Masking
class IFFM(nn.Module):
    def __init__(self):
        super().__init__()
        # DFFM의 초기 중요도 맵

    def normalize_0_1(self, iffm: torch.Tensor):
        min_val = iffm.min()
        max_val = iffm.max()
        return (iffm - min_val) / (max_val - min_val + 1e-8)

    def forward(self, x: torch.Tensor, grad_dict: dict, threshold=0.9, train=True):
        if train:
            return x

        loss_grad = grad_dict

        if loss_grad is None:
            return x

        # FFT 변환
        x_fft = fft.fft2(x)
        grad_fft = fft.fft2(loss_grad)

        # 2. 손실의 기울기를 기반으로 주파수 중요도 업데이트
        frequency_importance = torch.abs(grad_fft).mean(dim=0)
        dffm_norm = self.normalize_0_1(frequency_importance)
        # threshold_value = torch.quantile(dffm_norm.view(-1), 0.75)

        # 3. 주파수 마스크 생성
        mask = (dffm_norm <= threshold).float()

        # 4. 주파수 마스크 적용 및 역 FFT 변환
        x_fft_masked = x_fft * mask.to(x_fft.device)
        x_map_masked = torch.real(fft.ifft2(x_fft_masked))

        return x_map_masked

