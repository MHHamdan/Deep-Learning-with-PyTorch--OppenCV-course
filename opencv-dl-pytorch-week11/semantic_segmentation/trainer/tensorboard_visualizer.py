from torch.utils.tensorboard import SummaryWriter

from .visualizer import Visualizer


class TensorBoardVisualizer(Visualizer):
    def __init__(self):
        self._writer = SummaryWriter()

    def update_charts(self, train_metric, train_loss, test_metric, test_loss, learning_rate, epoch):
        if train_metric is not None:
            for metric_key, metric_value in train_metric.items():
                try:
                    iterator = iter(metric_value)
                    for idx, subvalue in enumerate(iterator):
                        self._writer.add_scalar("data/train_{}_{}".format(metric_key, idx), subvalue, epoch)
                except Exception as exception:   
                    self._writer.add_scalar("data/train_{}".format(metric_key), metric_value, epoch)
                    
                    
        if test_metric is not None:
            for test_metric_key, test_metric_value in test_metric.items():
                try:
                    iterator = iter(test_metric_value)
                    for idx, subvalue in enumerate(iterator):
                        self._writer.add_scalar("data/test_{}_{}".format(test_metric_key, idx), subvalue, epoch)
                except Exception as exception:   
                    self._writer.add_scalar("data/test_{}".format(test_metric_key), test_metric_value, epoch)
        
        if train_loss is not None:
            self._writer.add_scalar("data/train_loss", train_loss, epoch)
        if test_loss is not None:
            self._writer.add_scalar("data/test_loss", test_loss, epoch)

        if (train_loss is not None) and (test_loss is not None):
            self._writer.add_scalars("data/train-test-loss", {'train': train_loss, 'test': test_loss}, epoch)

        self._writer.add_scalar("data/learning_rate", learning_rate, epoch)

    def close_tensorboard(self):
        self._writer.close()
