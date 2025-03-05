import torch.nn as nn
import torchvision.models as models

from models.layers.attention import SpatialAttention


class ModifiedDensenetEncoder(nn.Module):
    def __init__(self, enc_model, cls_num):
        super().__init__()

        if enc_model == 'densenet121':
            encoder = models.densenet121(weights='IMAGENET1K_V1')
            self.emb_size = 256
        elif enc_model == 'densenet161':
            encoder = models.densenet161(weights='IMAGENET1K_V1')
            self.emb_size = 256
        elif enc_model == 'densenet169':
            encoder = models.densenet169(weights='IMAGENET1K_V1')
            self.emb_size = 256
        else:
            print("Error encoder models!!!, You can select a models in  this list ==> [densenet121, densenet161, densenet169]")
            print("Select default Model: [densenet121]")
            encoder = models.densenet121(weights='IMAGENET1K_V1')
            self.emb_size = 256

        self.stem = encoder.features[:4]
        self.sa_0 = SpatialAttention()

        self.dense_block_1 = encoder.features.denseblock1
        self.transition_1 = encoder.features.transition1
        self.sa_1 = SpatialAttention()

        self.dense_block_2 = encoder.features.denseblock2
        self.transition_2 = encoder.features.transition2
        self.sa_2 = SpatialAttention()

        self.conv_3x3 = nn.Conv2d(self.emb_size, self.emb_size, kernel_size=(3, 3), stride=1, padding=0)
        # self.conv_1x1 = nn.Conv2d(self.emb_size, 1, kernel_size=(1, 1), stride=1, padding=0)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(in_features=self.emb_size, out_features=cls_num)

    def forward(self, x):
        x_stem = self.stem(x)
        x_attn_0 = self.sa_0(x_stem)

        x_dense_1 = self.dense_block_1(x_attn_0)
        x_trans_1 = self.transition_1(x_dense_1)  # down-sizing
        x_attn_1 = self.sa_1(x_trans_1)

        # dense block 1
        x_dense_2 = self.dense_block_2(x_attn_1)
        x_trans_2 = self.transition_2(x_dense_2)  # down-sizing
        x_attn_2 = self.sa_2(x_trans_2)

        x_conv = self.conv_3x3(x_attn_2)
        x_pool = self.avg_pool(x_conv)

        x_pool = x_pool.view(x_pool.size(0), -1)
        result = self.classifier(x_pool)

        return result, [x_stem, x_dense_1, x_dense_2]
