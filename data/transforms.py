from torchvision import transforms
from data.frequency_mask import masked_image


def transform_handler(train=False, image_size=224, crop=False, jitter=False, noise=False, equalize=False,
                      bandpass=False):
    """
    PyTorch 데이터 변환(transform) 핸들러.

    Args:
        train (bool): 학습 모드 여부.
        image_size (int): 이미지 크기.
        crop (bool): 랜덤 크롭 적용 여부.
        jitter (bool): 밝기, 대비, 색상 변화 적용 여부.
        noise (bool): 가우시안 블러 적용 여부.
        equalize (bool): 히스토그램 평활화 적용 여부.
        bandpass (bool): 중~저주파 필터링 적용 여부.

    Returns:
        torchvision.transforms.Compose: 설정된 변환 객체.
    """
    assert (type(crop) is bool) or (type(jitter) is bool), "crop과 jitter의 value type은 bool만 가능합니다. [True or False]"

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if train:
        img_trans = []

        # 크롭 or Resize
        if crop:
            img_trans.append(transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)))
        else:
            img_trans.append(transforms.Resize((image_size, image_size)))

        # 밝기/색상 변화
        if jitter:
            img_trans.append(transforms.ColorJitter(brightness=(0.5, 0.9),
                                                    contrast=(0.4, 0.8),
                                                    saturation=(0.7, 0.9),
                                                    hue=(-0.2, 0.2)))
        # 블러 (노이즈)
        if noise:
            img_trans.append(transforms.GaussianBlur(kernel_size=(3, 3), sigma=(1.0, 2.0)))

        # 히스토그램 평활화 (Equalization)
        if equalize:
            img_trans.append(transforms.RandomEqualize(p=1))

        # 중~저주파 필터 적용
        if bandpass:
            img_trans.append(transforms.Lambda(lambda img: masked_image(img)))

        # 기본 변환 (수평 반전, 텐서 변환, 정규화)
        img_trans += [transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize(mean, std)]

        img_trans = transforms.Compose(img_trans)
        print(f'-------------------[Loaded train transform] [crop: {crop}, jitter: {jitter}, noise: {noise}, equalize: {equalize}, bandpass: {bandpass}]')

    else:
        img_trans = transforms.Compose([transforms.Resize((image_size, image_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])
        print('-------------------[Loaded transform] [Only Resizing]')

    return img_trans
