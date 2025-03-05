import os
import torch
import tqdm
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from models.builder import model_builder

from data.dataset import get_loader
from utils.config_functions import save_configs
from utils.metrics import Metrics
from utils.config_functions import get_configs


def get_lambda(epoch, total_epochs):
    p = epoch / total_epochs  # 학습 진행도 (0~1)
    return 2 / (1 + np.exp(-10 * p)) - 1


class Test:
    def __init__(self, cfg: dict, ep: int, ckp_fold: str):
        self.cfg = cfg
        self.ep = ep
        torch.manual_seed(self.cfg['seed'])
        torch.cuda.manual_seed_all(self.cfg['seed'])

        if self.cfg['do_logging']:
            os.makedirs(f"{self.cfg['log_path']}/tensorboard", exist_ok=True)
            self.summary = SummaryWriter(f"{self.cfg['log_path']}/tensorboard")
            save_configs(self.cfg)
        if self.cfg['do_ckp_save']:
            os.makedirs(f"{self.cfg['ckp_path']}", exist_ok=True)

        # ---------------------------------------------------------------------------
        # ------------------------------- Build Model -------------------------------
        # ---------------------------------------------------------------------------
        self.model = model_builder(self.cfg).to(self.cfg['device'])
        checkpoint = torch.load(f"{self.cfg['ckp_path']}/{ep}.pth", map_location=self.cfg['device'])
        print(f"{self.cfg['ckp_path']}/{ep}.pth")
        self.model.load_state_dict(checkpoint["model_state_dict"])

        print(f"-------------------[Loaded Model: {self.cfg['model']}]")
        self.metrics = Metrics(task="binary", num_classes=2)

        self.attack_power = {'Gamma': ['gamma_0.8', 'gamma_0.9', 'gamma_1.2'],
                             'Gaussian': ['blur_3', 'blur_9', 'blur_11'],
                             'JPEG': ['q_85', 'q_90', 'q_95'],
                             'Median': ['blur_3', 'blur_9', 'blur_11'],
                             'AHE': [False]}
        self.ckp_fold = ckp_fold

    def inference(self, loader, epoch, score_dict, desc):
        with torch.no_grad():
            self.model.eval()
            for _, (x, y) in enumerate(tqdm.tqdm(loader, desc=desc)):
                x = x.to(self.cfg['device'])
                y = y.to(self.cfg['device'])

                label_logit_source, _ = self.model(x)
                self.metrics.update(label_logit_source.argmax(1).cpu(), y.cpu())

            acc, apcer, bpcer, acer = self.metrics.cal_metrics()
            score_dict['epoch'] = epoch
            score_dict['acc'] = acc
            score_dict['apcer'] = apcer
            score_dict['bpcer'] = bpcer
            score_dict['acer'] = acer

            self.metrics.reset()

            return score_dict

    def run(self):
        attacks = ['JPEG', 'AHE']

        for attack in attacks:
            if self.ckp_fold == '1-fold':
                folder = 'A'
            else:
                folder = 'B'

            attack_power = self.attack_power[attack]
            for power in attack_power:
                attack_score = {'epoch': 0, 'acc': 0, 'apcer': 0, 'bpcer': 0, 'acer': 0}
                if power:
                    attack_path = f"E:/dataset/Ocular/Warsaw/PGLAV-GAN/Attack/{attack}/{folder}/{power}/fake"
                else:
                    attack_path = f"E:/dataset/Ocular/Warsaw/PGLAV-GAN/Attack/{attack}/{folder}/fake"

                attack_pp_loader = get_loader(train=False,
                                              image_size=self.cfg['image_size'],
                                              batch_size=self.cfg['batch_size'],
                                              fake_path=attack_path,
                                              live_path=self.cfg['te_live_dataset_path'])

                # Test Dataset
                test_score = self.inference(attack_pp_loader, self.ep, attack_score, desc=f"[{attack}/{power}/Test-->{self.ep}]")
                print(f"[{attack}/{power}-Test] APCER: {test_score['apcer'] * 100} | BPCER: {test_score['bpcer'] * 100} | ACER: {test_score['acer'] * 100}")


if __name__ == '__main__':
    folds = ['1-fold', '2-fold']
    ckp_ep = {'1-fold': 12, '2-fold': 23}

    for fold in folds:
        ep = ckp_ep[fold]
        cfg_path = f'E:/4th/backup/ablation/Warsaw/SwinTransformer/Bhattacharyya-4DFeature-AugBandPassMask-NotOtherAug/dist_0.01-label_1/Train_PGLAV-GAN/{fold}/log/train_parameters.yml'
        cfg = get_configs(cfg_path)

        test = Test(cfg, ep, fold)
        test.run()