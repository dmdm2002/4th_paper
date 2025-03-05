import os
import torch
import tqdm
import numpy as np

import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models.builder import model_builder

from runner.scheduler import CosineAnnealingWarmUpRestarts
# from runner.losses.mmd import cal_ditribution_loss
from runner.distribution_matching_loss import dist_matching_loss, fft_dist_matching_loss
from data.dataset import get_loader
from utils.config_functions import save_configs
from utils.metrics import Metrics
from utils.logger import logging_tensorboard, logging_txt


def get_lambda(epoch, total_epochs):
    p = epoch / total_epochs  # 학습 진행도 (0~1)
    return 2 / (1 + np.exp(-10 * p)) - 1


class Train:
    def __init__(self, cfg: dict):
        self.cfg = cfg
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
        print(f"-------------------[Loaded Model: {self.cfg['model']}]")

        # ----------------------------------------------------------------------------------------------
        # ------------------------------- Setting Train Param and Loader -------------------------------
        # ----------------------------------------------------------------------------------------------
        self.optimizer = optim.Adam(self.model.parameters(), self.cfg['lr'], (self.cfg['b1'], self.cfg['b2']))

        self.tr_loader = get_loader(train=True,
                                    image_size=self.cfg['image_size'],
                                    crop=self.cfg['crop'],
                                    jitter=self.cfg['jitter'],
                                    noise=self.cfg['noise'],
                                    equalize=self.cfg['equalize'],
                                    bandpass=self.cfg['bandpass'],
                                    batch_size=self.cfg['batch_size'],
                                    fake_path=self.cfg['tr_fake_dataset_path'],
                                    live_path=self.cfg['tr_live_dataset_path'],
                                    injection_data_path=self.cfg['tr_injection_fake_data_path'])

        self.te_loader = get_loader(train=False,
                                    image_size=self.cfg['image_size'],
                                    batch_size=self.cfg['batch_size'],
                                    fake_path=self.cfg['te_fake_dataset_path'],
                                    live_path=self.cfg['te_live_dataset_path'])

        self.attack_cross_spoofing_loader = get_loader(train=False,
                                                       image_size=self.cfg['image_size'],
                                                       batch_size=self.cfg['batch_size'],
                                                       fake_path=self.cfg['attack_cross_spoofing_path'],
                                                       live_path=self.cfg['te_live_dataset_path'])

        self.attack_cross_dataset_loader = get_loader(train=False,
                                                      image_size=self.cfg['image_size'],
                                                      batch_size=self.cfg['batch_size'],
                                                      fake_path=self.cfg['attack_cross_dataset_fake_path'],
                                                      live_path=self.cfg['attack_cross_dataset_live_path'])

        self.attack_pp_loader = get_loader(train=False,
                                           image_size=self.cfg['image_size'],
                                           batch_size=self.cfg['batch_size'],
                                           fake_path=self.cfg['attack_pp_path'],
                                           live_path=self.cfg['te_live_dataset_path'])

        self.scheduler = CosineAnnealingWarmUpRestarts(optimizer=self.optimizer, T_0=self.cfg['step_size'], T_mult=1,
                                                       eta_max=0.0001, T_up=len(self.tr_loader),
                                                       gamma=self.cfg['gamma'])

        self.loss_ce = nn.CrossEntropyLoss()
        self.metrics = Metrics(task="binary", num_classes=2)

        self.grad_dict = {layer: None for layer in self.cfg['grad_layer_list']}
        self.next_grad_dict = {layer: None for layer in self.cfg['grad_layer_list']}
        self.save_grad_hooks(self.cfg['grad_layer_list'])

    def grad_hook(self, name):
        def hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                target_grad = grad_output[0]
            else:
                target_grad = grad_output

            if self.next_grad_dict[name] is None:
                self.next_grad_dict[name] = target_grad.detach().sum(dim=0)
            else:
                self.next_grad_dict[name] += target_grad.detach().sum(dim=0)

        return hook

    def save_grad_hooks(self, layer_names):
        # 각 레이어에 개별적으로 hook을 등록
        for name in layer_names:
            layer = getattr(self.model, name)  # 해당 레이어 가져오기
            layer.register_full_backward_hook(self.grad_hook(name))

    def calculate_epoch_grad_average(self, total_data):
        for layer, grad_sum in self.next_grad_dict.items():
            if grad_sum is not None:
                # 배치 개수로 나누어 평균 기울기 계산
                self.grad_dict[layer] = grad_sum / total_data
                print(
                    f"Layer: {layer}, Grad Average Shape: {self.grad_dict[layer].shape}, Mean Value: {self.grad_dict[layer].mean().item()}")
            else:
                print(f"Warning: No gradients were captured for layer {layer}.")

    def inference(self, loader, epoch, score_dict, desc):
        with torch.no_grad():
            self.model.eval()
            for _, (x, y) in enumerate(tqdm.tqdm(loader, desc=desc)):
                x = x.to(self.cfg['device'])
                y = y.to(self.cfg['device'])

                label_logit_source, _ = self.model(x, grad_dict=self.grad_dict, train=False)
                self.metrics.update(label_logit_source.argmax(1).cpu(), y.cpu())

            acc, apcer, bpcer, acer = self.metrics.cal_metrics()
            score_dict['epoch'] = epoch
            score_dict['acc'] = acc
            score_dict['apcer'] = apcer
            score_dict['bpcer'] = bpcer
            score_dict['acer'] = acer

            self.metrics.reset()

            return score_dict

    def train_one_epoch(self, epoch, train_score):
        source_acc, target_acc = 0, 0
        label_loss_avg, domain_loss_avg, distribution_loss_avg, total_loss_avg = 0, 0, 0, 0

        for _, (x_source, x_target, y) in enumerate(
                tqdm.tqdm(self.tr_loader, desc=f"[{self.cfg['model']} Train-->{epoch}/{self.cfg['epoch']}]")):
            x_source = x_source.to(self.cfg['device'])
            x_target = x_target.to(self.cfg['device'])
            y = y.to(self.cfg['device'])

            label_logit_source,  dense_feature_source_list = self.model(x_source, grad_dict=self.grad_dict, train=True)
            label_logit_target,  dense_feature_target_list = self.model(x_target, grad_dict=self.grad_dict, train=True)

            source_label_loss = self.loss_ce(label_logit_source, y)
            target_label_loss = self.loss_ce(label_logit_target, y)
            label_loss = source_label_loss + target_label_loss

            dense_distribution_loss = fft_dist_matching_loss(dense_feature_source_list, dense_feature_target_list, loss_type='Bhattacharyya')

            total_loss = (label_loss * 0.9) + (dense_distribution_loss * 0.1)

            source_acc += (label_logit_source.argmax(1) == y).type(torch.float).sum().item()
            target_acc += (label_logit_target.argmax(1) == y).type(torch.float).sum().item()

            label_loss_avg += label_loss * 0.9
            distribution_loss_avg += dense_distribution_loss * 0.1
            total_loss_avg += total_loss

            # 역전파
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        train_score['epoch'] = epoch
        train_score['source_acc'] = source_acc / len(self.tr_loader.dataset)
        train_score['target_acc'] = target_acc / len(self.tr_loader.dataset)

        train_score['label_loss_avg'] = label_loss_avg / len(self.tr_loader)
        # train_score['domain_loss_avg'] = domain_loss_avg / len(self.tr_loader)
        train_score['distribution_loss_avg'] = distribution_loss_avg / len(self.tr_loader)
        train_score['total_loss'] = total_loss_avg / len(self.tr_loader)

        return train_score

    def run(self):
        train_score = {'epoch': 0, 'source_acc': 0, 'target_acc': 0, 'label_loss_avg': 0, 'domain_loss_avg': 0,
                       'distribution_loss_avg': 0, 'total_loss': 0}
        test_score = {'epoch': 0, 'acc': 0, 'apcer': 0, 'bpcer': 0, 'acer': 0}
        best_score = {'epoch': 0, 'acc': 0, 'apcer': 0, 'bpcer': 0, 'acer': 0}
        attack_cross_spoofing_score = {'epoch': 0, 'acc': 0, 'apcer': 0, 'bpcer': 0, 'acer': 0}
        attack_pp_score = {'epoch': 0, 'acc': 0, 'apcer': 0, 'bpcer': 0, 'acer': 0}
        attack_cross_db_score = {'epoch': 0, 'acc': 0, 'apcer': 0, 'bpcer': 0, 'acer': 0}

        total_data = len(self.tr_loader.dataset)
        for epoch in range(self.cfg['epoch']):
            self.model.train()
            # 매 epoch마다 grad_dict 초기화 (매 epoch 평균을 구하려면 누적 가능)
            self.calculate_epoch_grad_average(total_data)

            train_score = self.train_one_epoch(epoch, train_score)
            self.scheduler.step(epoch)

            # Test Dataset
            test_score = self.inference(self.te_loader, epoch, test_score, desc=f"[Test-->{epoch}/{self.cfg['epoch']}]")

            if best_score['acc'] <= test_score['acc']:
                best_score['epoch'] = test_score['epoch']
                best_score['acc'] = test_score['acc']
                best_score['apcer'] = test_score['apcer']
                best_score['bpcer'] = test_score['bpcer']
                best_score['acer'] = test_score['acer']

                attack_cross_spoofing_score = self.inference(self.attack_cross_spoofing_loader, epoch,
                                                             attack_cross_spoofing_score,
                                                             desc=f"[Attack Cross Spoofing-->{epoch}/{self.cfg['epoch']}]")
                attack_pp_score = self.inference(self.attack_pp_loader, epoch, attack_pp_score,
                                                 desc=f"[Attack Post-processing-->{epoch}/{self.cfg['epoch']}]")
                attack_cross_db_score = self.inference(self.attack_cross_dataset_loader, epoch, attack_cross_db_score,
                                                       desc=f"[Attack Cross Dataset-->{epoch}/{self.cfg['epoch']}]")

            if self.cfg['do_logging_tensorboard']:
                logging_tensorboard(self.summary, train_score, epoch, cate='Train')
                logging_tensorboard(self.summary, test_score, epoch, cate='Test')
                logging_tensorboard(self.summary, attack_cross_spoofing_score, epoch, cate='Attack Cross Spoofing')
                logging_tensorboard(self.summary, attack_cross_db_score, epoch, cate='Attack Cross Dataset')
                logging_tensorboard(self.summary, attack_pp_score, epoch, cate='Attack Post-processing')

            if self.cfg['do_logging_txt']:
                f = open(f"{self.cfg['log_path']}/ACC_LOSS_LOG.txt", 'a', encoding='utf-8')
                logging_txt(train_score, f, cate='Train')
                logging_txt(test_score, f, cate='Test')
                logging_txt(attack_cross_spoofing_score, f, cate='Attack Cross Spoofing')
                logging_txt(attack_cross_db_score, f, cate='Attack Cross Dataset')
                logging_txt(attack_pp_score, f, cate='Attack Post-processing')
                f.close()

            if self.cfg['do_print']:
                print('\n')
                print(
                    '=====================================================================================================')
                print(f"Epoch: {epoch}/{self.cfg['epoch']}")
                print(
                    f"Train Source Acc: {train_score['source_acc']} | Target Acc: {train_score['target_acc']} | Total Loss: {train_score['total_loss']} | Domain Loss: {train_score['domain_loss_avg']} | Distribution Loss: {train_score['distribution_loss_avg']}| Label Loss: {train_score['label_loss_avg']}")
                print(
                    f"Test  APCER: {test_score['apcer'] * 100} | BPCER: {test_score['bpcer'] * 100} | ACER: {test_score['acer'] * 100}")
                print(
                    f"-----------------------------------[Best Epoch {best_score['epoch']}]-----------------------------------")
                print(
                    f"[Best Test] APCER: {best_score['apcer'] * 100} | BPCER: {best_score['bpcer'] * 100} | ACER: {best_score['acer'] * 100}")
                print(
                    f"[Best Cross Spoofing] APCER: {attack_cross_spoofing_score['apcer'] * 100} | BPCER: {attack_cross_spoofing_score['bpcer'] * 100} | ACER: {attack_cross_spoofing_score['acer'] * 100}")
                print(
                    f"[Best Cross Dataset] APCER: {attack_cross_db_score['apcer'] * 100} | BPCER: {attack_cross_db_score['bpcer'] * 100} | ACER: {attack_cross_db_score['acer'] * 100}")
                print(
                    f"[Best AHE] APCER: {attack_pp_score['apcer'] * 100} | BPCER: {attack_pp_score['bpcer'] * 100} | ACER: {attack_pp_score['acer'] * 100}")
                print(
                    '=====================================================================================================')

            if self.cfg['do_ckp_save']:
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "AdamW_state_dict": self.optimizer.state_dict(),
                        "epoch": epoch,
                    },
                    os.path.join(f"{self.cfg['ckp_path']}/", f"{epoch}.pth"),
                )
