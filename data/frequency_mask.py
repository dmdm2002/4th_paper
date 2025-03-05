import torch
import numpy as np
from PIL import Image


def band_pass_mask(shape, low_cut=0.05, high_cut=0.3):
    """
    중~저주파만 남기는 마스크 생성 함수 (고주파 제거).

    Args:
        shape (tuple): (H, W) 이미지 크기.
        low_cut (float): 0~1 사이 값, 유지할 저주파 비율.
        high_cut (float): 0~1 사이 값, 제거할 고주파 비율.

    Returns:
        mask (numpy.ndarray): 2D 주파수 마스크.
    """
    H, W = shape
    center_x, center_y = W // 2, H // 2  # 주파수 중심 좌표
    mask = np.zeros((H, W), np.float32)  # 기본적으로 모든 주파수 차단

    # 유지할 주파수 영역 설정
    low_radius = int(min(H, W) * low_cut)  # 저주파 유지 반경
    high_radius = int(min(H, W) * high_cut)  # 고주파 차단 반경

    # 중~저주파 영역 유지
    mask[center_y - high_radius:center_y + high_radius, center_x - high_radius:center_x + high_radius] = 1
    mask[center_y - low_radius:center_y + low_radius, center_x - low_radius:center_x + low_radius] = 1

    return mask

def masked_image(image: Image.Image) -> Image.Image:
    """
    고주파를 제거하고 중~저주파만 남기는 변환.

    Args:
        image (PIL.Image): 입력 이미지

    Returns:
        PIL.Image: 변환된 이미지
    """
    # **Step 1: PIL → NumPy 변환 (필수, FFT 수행을 위해)**
    image_np = np.array(image.convert("L"))  # Grayscale 변환 후 NumPy 배열 (H, W)

    # **Step 2: FFT 변환 후 고주파 마스킹 적용**
    fft_image = np.fft.fftshift(np.fft.fft2(image_np))
    mask = band_pass_mask(image_np.shape, low_cut=0.05, high_cut=0.3)  # 중~저주파 유지
    fft_filtered = fft_image * mask  # 마스킹 적용

    # **Step 3: IFFT 변환 (이미지 복원)**
    image_filtered = np.fft.ifft2(np.fft.ifftshift(fft_filtered)).real
    image_filtered = np.clip(image_filtered, 0, 255).astype(np.uint8)  # 정규화 및 uint8 변환

    # **Step 4: NumPy → PIL 변환**
    image_pil = Image.fromarray(image_filtered).convert("RGB")  # 다시 3채널로 변환

    return image_pil  # PIL.Image 반환
