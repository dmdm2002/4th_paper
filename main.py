from utils.config_functions import get_configs
# from runner.train import Train
from runner.train_iffm import Train
# from runner.test import Test
import argparse
import torch
import gc
import sys

# print(os.path.abspath(__file__))
if __name__ == '__main__':
    torch.cuda.empty_cache()
    gc.collect()

    parser = argparse.ArgumentParser(description='Proposed Model Ablation!!')
    parser.add_argument('--path', type=str, required=True, help='이미지 폴더 경로를 입력하세요')
    parser.add_argument('--model', type=str, required=True, help='모델 이름')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset Name')
    parser.add_argument('--backbone_model', type=str, required=True, help='백본 모델')
    parser.add_argument('--fold', type=str, required=True, help='Train Fold')

    args = parser.parse_args()
    # print(f'Now Fold: {args.fold}')

    cfg = get_configs('config.yml')
    cfg['model'] = f'{args.model}'
    cfg['backbone_model'] = f'{args.backbone_model}'
    cfg['ckp_path'] = f'{args.path}/ckp'
    cfg['log_path'] = f'{args.path}/log'

    if args.dataset == 'ND':
        cross_db = 'Warsaw'
    else:
        cross_db = 'ND'

    if args.fold == '1-fold':
        cfg['tr_fake_dataset_path'] = f"E:/dataset/Ocular/{args.dataset}/PGLAV-GAN/1-fold/B/fake"
        cfg['tr_injection_fake_data_path'] = f"E:/dataset/Ocular/{args.dataset}/RandomInjections/PGLAV-GAN-RandomPatchInjectionAug/1-fold/B/fake"
        cfg['tr_live_dataset_path'] = f"E:/dataset/Ocular/{args.dataset}/original/B"

        cfg['te_fake_dataset_path'] = f"E:/dataset/Ocular/{args.dataset}/PGLAV-GAN/2-fold/A/fake"
        cfg['te_live_dataset_path'] = f"E:/dataset/Ocular/{args.dataset}/original/A"

        cfg['attack_cross_dataset_fake_path'] = f"E:/dataset/Ocular/{cross_db}/DDIM/2-fold/A/fake"
        cfg['attack_cross_dataset_live_path'] = f"E:/dataset/Ocular/{cross_db}/original/A"

        cfg['attack_cross_spoofing_path'] = f"E:/dataset/Ocular/{args.dataset}/UVC_GAN/2-fold/A/fake"
        cfg['attack_pp_path'] = f"E:/dataset/Ocular/{args.dataset}/PGLAV-GAN/Attack/AHE/A/fake"

    else:
        cfg['tr_fake_dataset_path'] = f"E:/dataset/Ocular/{args.dataset}/PGLAV-GAN/2-fold/A/fake"
        cfg['tr_injection_fake_data_path'] = f"E:/dataset/Ocular/{args.dataset}/RandomInjections/PGLAV-GAN-RandomPatchInjectionAug/2-fold/A/fake"
        cfg['tr_live_dataset_path'] = f"E:/dataset/Ocular/{args.dataset}/original/A"

        cfg['te_fake_dataset_path'] = f"E:/dataset/Ocular/{args.dataset}/PGLAV-GAN/1-fold/B/fake"
        cfg['te_live_dataset_path'] = f"E:/dataset/Ocular/{args.dataset}/original/B"

        # cfg['attack_fake_DDPM_dataset_path'] = f"E:/dataset/Ocular/{args.dataset}/Ocular/DDPM/1-fold/B/fake"
        cfg['attack_cross_dataset_fake_path'] = f"E:/dataset/Ocular/{cross_db}/DDIM/1-fold/B/fake"
        cfg['attack_cross_dataset_live_path'] = f"E:/dataset/Ocular/{cross_db}/original/B"

        cfg['attack_cross_spoofing_path'] = f"E:/dataset/Ocular/{args.dataset}/UVC_GAN/1-fold/B/fake"
        cfg['attack_pp_path'] = f"E:/dataset/Ocular/{args.dataset}/PGLAV-GAN/Attack/AHE/B/fake"

    print(f"Start Train: [DB: {args.dataset}]")
    cls_train = Train(cfg)
    cls_train.run()

    sys.exit()