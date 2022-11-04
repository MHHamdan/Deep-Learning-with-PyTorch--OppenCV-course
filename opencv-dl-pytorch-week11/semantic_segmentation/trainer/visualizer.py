from abc import ABC, abstractmethod


class Visualizer(ABC):
    @abstractmethod
    def update_charts(self, train_metric, train_loss, test_metric, test_loss, learning_rate, epoch):
        pass
