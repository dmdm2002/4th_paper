import os
import gc
import time
import torch
from numba import cuda

# fkd_lambda = [0.05, 0.1, 0.2, 0.3]
# cls_lambda = [0.95, 0.9, 0.8, 0.7]

db = ['Warsaw']
folds = ['1-fold', '2-fold']
models = ['IFFMSwinTransformer'] # DESAResNetEncoder DESADenseNetEncoder
backbone_model = 'swin_base'

for db_name in db:
    for model in models:
        for fold in folds:
            device = cuda.get_current_device()
            device.reset()

            torch.cuda.empty_cache()
            gc.collect()

            path = f'E:/4th/backup/Ablation/{db_name}/{model}/{backbone_model}/WeightSharing-IFFM-FFTBhattacharyya-AugBandPassMask/dist_0.1-label_0.9/Train_PGLAV-GAN/{fold}/'
            os.system(f'python ./main.py --path={path} --model={model} --dataset={db_name} --backbone_model={backbone_model} --fold={fold}')

            time.sleep(3)

