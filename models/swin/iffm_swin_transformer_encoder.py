import timm
import torch
import torch.nn as nn

from models.layers.iffm import IFFM


class IFFMSwinEncoder(nn.Module):
    def __init__(self, enc_model: str, cls_num: int):
        super().__init__()
        enc_model = enc_model.lower()

        if enc_model == 'swin_tiny':
            encoder = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=cls_num)
            self.emb_size = 1024
        elif enc_model == 'swin_small':
            encoder = timm.create_model('swin_small_patch4_window7_224', pretrained=True, num_classes=cls_num)
            self.emb_size = 1024
        elif enc_model == 'swin_base':
            encoder = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=cls_num)
            self.emb_size = 1024
        elif enc_model == 'swin_large':
            encoder = timm.create_model('swin_large_patch4_window7_224', pretrained=True, num_classes=cls_num)
            self.emb_size = 1024
        else:
            print(
                "Error encoder models!!!, You can select a models in  this list ==> [swin_tiny, swin_small, swin_base, swin_large]")
            print("Select default Model: [swin_base]")
            encoder = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=cls_num)
            self.emb_size = 1024

        self.iffm = IFFM()
        self.patch_embed = encoder.patch_embed

        self.swin_stage_0 = encoder.layers[0]
        self.swin_stage_1 = encoder.layers[1]
        self.swin_stage_2 = encoder.layers[2]
        self.swin_stage_3 = encoder.layers[3]

        self.norm = encoder.norm
        self.head = encoder.head


    def forward(self, x, grad_dict: dict, train=True):
        x_p_embed = self.patch_embed(x)

        x_swin_0 = self.swin_stage_0(x_p_embed)
        x_iffm_0 = self.iffm(x_swin_0, grad_dict['swin_stage_0'], train=train)

        x_swin_1 = self.swin_stage_1(x_iffm_0)
        x_iffm_1 = self.iffm(x_swin_1, grad_dict['swin_stage_1'], train=train)

        x_swin_2 = self.swin_stage_2(x_iffm_1)
        x_iffm_2 = self.iffm(x_swin_2, grad_dict['swin_stage_2'], train=train)

        x_swin_3 = self.swin_stage_3(x_iffm_2)

        x_nrom = self.norm(x_swin_3)
        out = self.head(x_nrom)

        return out, [x_swin_0, x_swin_1, x_swin_2, x_swin_3]

