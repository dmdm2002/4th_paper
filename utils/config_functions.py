import yaml
import os


def get_configs(path):
    assert os.path.exists(path), f"경로[{path}]에 해당 파일이 존재하지 않습니다."
    with open(path) as f:
        config = yaml.safe_load(f)

    return config


def save_configs(cfg):
    with open(f"{cfg['log_path']}/train_parameters.yml", "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

