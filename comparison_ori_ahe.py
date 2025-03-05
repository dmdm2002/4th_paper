import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드 (흑백)
image_path1 = 'E:/dataset/Ocular/Warsaw/PGLAV-GAN/1-fold/B/fake/0040_REAL_L_1.png'  # Original
image_path2 = 'E:/dataset/Ocular/Warsaw/original/B/0040/0040_REAL_L_1.png'  # Live

image = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

# 이미지가 로드되지 않았을 경우 예외 처리
if image is None or image2 is None:
    raise FileNotFoundError("이미지 파일을 찾을 수 없습니다. 경로를 확인하세요.")

# 1. 크기 조정 (모든 이미지를 동일한 크기로 맞춤)
image = cv2.resize(image, (224, 224))
image2 = cv2.resize(image2, (224, 224))

# 2. FFT 변환 (CLAHE 적용 전)
fft_original = np.fft.fftshift(np.fft.fft2(image))
magnitude_original = np.abs(fft_original)

# 3. CLAHE 적용
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(3,3))
image_clahe = clahe.apply(image)

# 4. CLAHE 적용 후 FFT 변환
fft_clahe = np.fft.fftshift(np.fft.fft2(image_clahe))
magnitude_clahe = np.abs(fft_clahe)

# 5. Live 이미지(원본) FFT 변환
fft_live = np.fft.fftshift(np.fft.fft2(image2))
magnitude_live = np.abs(fft_live)

# 6. 주파수 차이 계산
diff_live_original = np.abs(magnitude_live - magnitude_original)  # Live - Original
diff_live_clahe = np.abs(magnitude_live - magnitude_clahe)  # Live - CLAHE

# 7. 주파수 스펙트럼 비교 시각화
plt.figure(figsize=(18, 10))

plt.subplot(2, 3, 1)
plt.title("FFT Magnitude (Original)")
plt.imshow(np.log(1 + magnitude_original), cmap='gray')

plt.subplot(2, 3, 2)
plt.title("FFT Magnitude (CLAHE Applied)")
plt.imshow(np.log(1 + magnitude_clahe), cmap='gray')

plt.subplot(2, 3, 3)
plt.title("FFT Magnitude (Live)")
plt.imshow(np.log(1 + magnitude_live), cmap='gray')

plt.subplot(2, 3, 4)
plt.title("Live - Original (Frequency Difference)")
plt.imshow(np.log(1 + diff_live_original), cmap='jet')

plt.subplot(2, 3, 5)
plt.title("Live - CLAHE (Frequency Difference)")
plt.imshow(np.log(1 + diff_live_clahe), cmap='jet')

plt.show()
