from torch.utils.tensorboard import SummaryWriter


def logging_tensorboard(summary: SummaryWriter, log_dict: dict, ep, cate='Train'):
    key_list = list(log_dict.keys())
    print(f'-------------------[Writing a {cate} log on TensorBoard]')

    for key in key_list:
        if key != 'epoch':
            summary.add_scalar(f'{cate}/{key}', log_dict[key], ep)


def logging_txt(log_dict, f, cate='Train'):
    key_list = list(log_dict)

    f.write(f"[{cate} ")
    for key in key_list:
        f.write(f"{key}: {log_dict[key]}\t")
    f.write(f"]\n")

    return f

