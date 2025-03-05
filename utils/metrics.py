from torchmetrics.classification import ConfusionMatrix


class Metrics:
    def __init__(self, task, num_classes):
        super().__init__()
        self.conf_mat = ConfusionMatrix(task=task, num_classes=num_classes)

    def update(self, logit, y):
        self.conf_mat.update(logit, y)

    def cal_metrics(self):
        [[tn, fp], [fn, tp]] = self.conf_mat.compute()
        acc = (tp + tn) / (tp + fn + fp + tn) if (tp + fn + fp + tn) != 0 else 0
        apcer = fp / (tn + fp) if (tn + fp) != 0 else 0
        bpcer = fn / (fn + tp) if (fn + tp) != 0 else 0
        acer = (apcer + bpcer) / 2

        return acc, apcer, bpcer, acer

    def reset(self):
        self.conf_mat.reset()
