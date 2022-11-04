from abc import ABC, abstractmethod


class BaseMetric(ABC):
    @abstractmethod
    def update_value(self, output, target):
        pass

    @abstractmethod
    def get_metric_value(self):
        pass

    @abstractmethod
    def reset(self):
        pass
