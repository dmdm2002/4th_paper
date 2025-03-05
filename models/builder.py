from models import densenet, swin


def model_builder(cfg: dict):
    # ---------------------------------------------------------------------------
    # ------------------------------- Build Model -------------------------------
    # ---------------------------------------------------------------------------
    if cfg['model'] == 'DensenetEncoder':
        return densenet.DensenetEncoder(cfg['backbone_model'], cfg['cls_num'])
    elif cfg['model'] == 'DensenetGRLEncoder':
        return densenet.DensenetGRLEncoder(cfg['backbone_model'], cfg['cls_num'])
    elif cfg['model'] == 'ModifiedDensenetEncoder':
        return densenet.ModifiedDensenetEncoder(cfg['backbone_model'], cfg['cls_num'])
    elif cfg['model'] == 'SwinTransformer':
        return swin.SwinEncoder(cfg['backbone_model'], cfg['cls_num'])
    elif cfg['model'] == 'IFFMSwinTransformer':
        return swin.IFFMSwinEncoder(cfg['backbone_model'], cfg['cls_num'])
    else:
        print('지원하지 않는 모델입니다.')
        exit()
