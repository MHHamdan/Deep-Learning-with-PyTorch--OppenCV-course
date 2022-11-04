import torch

from .utils import AverageMeter
from .voc_eval import VOCEvaluator
from .base_metric import BaseMetric


class AccuracyEstimator(BaseMetric):
    def __init__(self, topk=(1, )):
        self.topk = topk
        self.metrics = [AverageMeter() for i in range(len(topk) + 1)]

    def reset(self):
        for i in range(len(self.metrics)):
            self.metrics[i].reset()

    def update_value(self, pred, target):
        """Computes the precision@k for the specified values of k"""
        with torch.no_grad():
            maxk = max(self.topk)
            batch_size = target.size(0)

            _, pred = pred.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            for i, k in enumerate(self.topk):
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                self.metrics[i].update(correct_k.mul_(100.0 / batch_size).item())

    def get_metric_value(self):
        metrics = {}
        for i, k in enumerate(self.topk):
            metrics["top{}".format(k)] = self.metrics[i].avg
        return metrics


class APEstimator(BaseMetric):
    def __init__(self, classes):
        self.classes = classes
        self.metrics = AverageMeter()
        self.evaluator = None

    def reset(self):
        self.metrics.reset()
        self.evaluator = VOCEvaluator(self.classes)

    def update_value(self, pred, target):
        """Computes AP
        """
        self.evaluator.add_sample(pred, target)

    def calculate_value(self):
        """Computes AP
        """
        aps = self.evaluator.evaluate()
        for class_idx in range(len(self.classes)):
            if self.classes[class_idx] == '__background__':
                continue
            if len(aps) > class_idx - 1:
                self.metrics.update(aps[class_idx - 1])
            else:
                self.metrics.update(0.0)

    def get_metric_value(self):
        metrics = {}
        metrics["mAP"] = self.metrics.avg
        return metrics
