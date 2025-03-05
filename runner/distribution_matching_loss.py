from runner.losses import MMDLoss, MahalanobisLoss, BhattacharyyaLoss
import torch
import torch.fft as fft


def fft_dist_matching_loss(source_list, target_list, loss_type='MMD'):
    full_coral_loss = 0

    if loss_type == 'MMD':
        loss_func = MMDLoss()
    elif loss_type == 'Mahalanobis':
        loss_func = MahalanobisLoss()
    elif loss_type == 'Bhattacharyya':
        loss_func = BhattacharyyaLoss()
    else:
        print(f'해당 [Loss :{loss_type}] 는 지원하지 않습니다.')
        print(f'Default Loss : MMDLoss 를 계산합니다.')
        loss_func = MMDLoss()

    for _, (source, target) in enumerate(zip(source_list, target_list)):
        source = fft.fft2(source)
        source = torch.abs(source)

        target = fft.fft2(target)
        target = torch.abs(target)

        # source = torch.mean(source, dim=[2, 3])  # (batch, feature_dim)
        # target = torch.mean(target, dim=[2, 3])

        loss = loss_func(source, target)
        full_coral_loss += loss

    return full_coral_loss / len(source_list)

def dist_matching_loss(source_list, target_list, loss_type='MMD'):
    full_coral_loss = 0

    if loss_type == 'MMD':
        loss_func = MMDLoss()
    elif loss_type == 'Mahalanobis':
        loss_func = MahalanobisLoss()
    elif loss_type == 'Bhattacharyya':
        loss_func = BhattacharyyaLoss()
    else:
        print(f'해당 [Loss :{loss_type}] 는 지원하지 않습니다.')
        print(f'Default Loss : MMDLoss 를 계산합니다.')
        loss_func = MMDLoss()

    for _, (source, target) in enumerate(zip(source_list, target_list)):
        loss = loss_func(source, target)
        full_coral_loss += loss

    return full_coral_loss / len(source_list)
