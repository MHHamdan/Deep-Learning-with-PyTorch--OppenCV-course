from collections import defaultdict

import matplotlib.pyplot as plt

from .visualizer import Visualizer


class MatplotlibVisualizer(Visualizer):
    def __init__(self):
        self._epochs = []
        self._metrics = defaultdict(list)
        self._figures = {}
        self._axes = {}

    def init_new_figure(self, name):
        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel("Epochs")
        ax.set_ylabel(name)
        self._figures[name] = fig
        self._axes[name] = ax

    def plot(self):
        for key, value in self._metrics.items():
            if key not in self._figures:
                self.init_new_figure(key)
            ax = self._axes[key]
            fig = self._figures[key]
            if ax.lines:
                ax.lines[0].set_xdata(self._epochs)
                ax.lines[0].set_ydata(value)
            else:
                ax.plot(self._epochs, value)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
        plt.show()

    def _update_metrics(self, name, value):
        self._metrics[name].append(value)

    def update_charts(self, train_metric, train_loss, test_metric, test_loss, learning_rate, epoch):
        if train_metric is not None:
            for metric_key, metric_value in train_metric.items():
                try:
                    iterator = iter(metric_value)
                    for idx, subvalue in enumerate(iterator):
                        self._update_metrics("train_{}_{}".format(metric_key, idx), subvalue)
                except TypeError as _:
                    self._update_metrics("train_{}".format(metric_key), metric_value)

        if test_metric is not None:
            for metric_key, metric_value in test_metric.items():
                try:
                    iterator = iter(metric_value)
                    for idx, subvalue in enumerate(iterator):
                        self._update_metrics("test_{}_{}".format(metric_key, idx), subvalue)
                except TypeError as _:
                    self._update_metrics("test_{}".format(metric_key), metric_value)

        if train_loss is not None:
            self._update_metrics("train_loss", train_loss)
        if test_loss is not None:
            self._update_metrics("test_loss", test_loss)

        self._update_metrics("learning_rate", learning_rate)
        self._epochs.append(epoch)
        self.plot()
