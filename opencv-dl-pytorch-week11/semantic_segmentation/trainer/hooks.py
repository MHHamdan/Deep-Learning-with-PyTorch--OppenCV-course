"""Implementation of several hooks that used in a Trainer class."""
from operator import itemgetter

import torch

from tqdm.auto import tqdm

from .utils import AverageMeter
from .encoder import DataEncoder


def train_hook_default(
    model,
    loader,
    loss_fn,
    optimizer,
    device,
    data_getter=itemgetter("image"),
    target_getter=itemgetter("mask"),
    iterator_type=tqdm,
    prefix="",
    stage_progress=False
):
    """ Default train loop function.

    Arguments:
        model (nn.Module): torch model which will be train.
        loader (torch.utils.DataLoader): dataset loader.
        loss_fn (callable): loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (str): Specifies device at which samples will be uploaded.
        data_getter (Callable): function object to extract input data from the sample prepared by dataloader.
        target_getter (Callable): function object to extract target data from the sample prepared by dataloader.
        iterator_type (iterator): type of the iterator.
        prefix (string): prefix which will be add to the description string.
        stage_progress (bool): if True then progress bar will be show.

    Returns:
        Dictionary of output metrics with keys:
            loss: average loss.
    """
    model = model.train()
    iterator = iterator_type(loader, disable=not stage_progress, dynamic_ncols=True)
    loss_avg = AverageMeter()
    for i, sample in enumerate(iterator):
        optimizer.zero_grad()
        inputs = data_getter(sample).to(device)
        targets = target_getter(sample).to(device)
        predicts = model(inputs)
        loss = loss_fn(predicts, targets)
        loss.backward()
        optimizer.step()
        loss_avg.update(loss.item())
        status = "{0}[Train][{1}] Loss_avg: {2:.5}, Loss: {3:.5}, LR: {4:.5}".format(
            prefix, i, loss_avg.avg, loss_avg.val, optimizer.param_groups[0]["lr"]
        )
        iterator.set_description(status)
    return {"loss": loss_avg.avg}


def train_hook_detection(
    model,
    loader,
    loss_fn,
    optimizer,
    device,
    data_getter=None,  # pylint: disable=W0613
    target_getter=None,  # pylint: disable=W0613
    iterator_type=tqdm,
    prefix="",
    stage_progress=False
):
    """ Default train loop function.

    Arguments:
        model (nn.Module): torch model which will be train.
        loader (torch.utils.DataLoader): dataset loader.
        loss_fn (callable): loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (str): Specifies device at which samples will be uploaded.
        data_getter (Callable): function object to extract input data from the sample prepared by dataloader.
        target_getter (Callable): function object to extract target data from the sample prepared by dataloader.
        iterator_type (iterator): type of the iterator.
        prefix (string): prefix which will be add to the description string.
        stage_progress (bool): if True then progress bar will be show.

    Returns:
        Dictionary of output metrics with keys:
            loss: average loss.
    """
    model = model.train()
    iterator = iterator_type(loader, disable=not stage_progress, dynamic_ncols=True)
    loss_avg = AverageMeter()
    for i, sample in enumerate(iterator):
        optimizer.zero_grad()
        inputs = sample[0].to(device)
        target_boxes = sample[1].to(device)
        target_labels = sample[2].to(device)
        pred_boxes, pred_labels = model(inputs)
        loc_loss, cls_loss = loss_fn(pred_boxes, target_boxes, pred_labels, target_labels)
        loss = loc_loss + cls_loss
        loss.backward()
        optimizer.step()
        loss_avg.update(loss.item())
        status = "{0}[Train][{1}] Loss_avg: {2:.5}, Loss: {3:.5}, Loc_loss: {4:.5}, Cls_loss: {5:.5}, LR: {6:.5}".format(
            prefix, i, loss_avg.avg, loss_avg.val, loc_loss, cls_loss, optimizer.param_groups[0]["lr"]
        )
        iterator.set_description(status)
    return {"loss": loss_avg.avg}


def test_hook_default(
    model,
    loader,
    loss_fn,
    metric_fn,
    device,
    data_getter=itemgetter("image"),
    target_getter=itemgetter("mask"),
    iterator_type=tqdm,
    prefix="",
    stage_progress=False,
    get_key_metric=itemgetter("accuracy")
):
    """ Default test loop function.

    Arguments:
        model (nn.Module): torch model which will be train.
        loader (torch.utils.DataLoader): dataset loader.
        loss_fn (callable): loss function.
        metric_fn (callable): evaluation metric function.
        device (str): Specifies device at which samples will be uploaded.
        data_getter (Callable): function object to extract input data from the sample prepared by dataloader.
        target_getter (Callable): function object to extract target data from the sample prepared by dataloader.
        iterator_type (iterator): type of the iterator.
        prefix (string): prefix which will be add to the description string.
        stage_progress (bool): if True then progress bar will be show.

    Returns:
        Dictionary of output metrics with keys:
            metric: output metric.
            loss: average loss.
    """
    model = model.eval()
    iterator = iterator_type(loader, disable=not stage_progress, dynamic_ncols=True)
    loss_avg = AverageMeter()
    metric_fn.reset()
    for i, sample in enumerate(iterator):
        inputs = data_getter(sample).to(device)
        targets = target_getter(sample).to(device)
        with torch.no_grad():
            predict = model(inputs)
            loss = loss_fn(predict, targets)
        loss_avg.update(loss.item())
        predict = predict.softmax(dim=1).detach()
        metric_fn.update_value(predict, targets)
        status = "{0}[Test][{1}] Loss_avg: {2:.5}".format(prefix, i, loss_avg.avg)
        if get_key_metric is not None:
            status = status + ", Metric_avg: {0:.5}".format(get_key_metric(metric_fn.get_metric_value()))
        iterator.set_description(status)
    output = {"metric": metric_fn.get_metric_value(), "loss": loss_avg.avg}
    return output


def test_hook_detection(
    model,
    loader,
    loss_fn,
    metric_fn,
    device,
    data_getter=None,  # pylint: disable=W0613
    target_getter=None,  # pylint: disable=W0613
    iterator_type=tqdm,
    prefix="",
    stage_progress=False,
    get_key_metric=itemgetter("AP")
):
    """ Default test loop function.

    Arguments:
        model (nn.Module): torch model which will be train.
        loader (torch.utils.DataLoader): dataset loader.
        loss_fn (callable): loss function.
        metric_fn (callable): evaluation metric function.
        device (str): Specifies device at which samples will be uploaded.
        data_getter (Callable): function object to extract input data from the sample prepared by dataloader.
        target_getter (Callable): function object to extract target data from the sample prepared by dataloader.
        iterator_type (iterator): type of the iterator.
        prefix (string): prefix which will be add to the description string.
        stage_progress (bool): if True then progress bar will be show.

    Returns:
        Dictionary of output metrics with keys:
            metric: output metric.
            loss: average loss.
    """
    model = model.eval()
    iterator = iterator_type(loader, disable=not stage_progress, dynamic_ncols=True)
    loss_avg = AverageMeter()
    metric_fn.reset()
    encoder = DataEncoder((300, 300))

    for i, sample in enumerate(iterator):
        inputs = sample[0].to(device)
        target_boxes = sample[1].to(device)
        target_labels = sample[2].to(device)

        batch_size = inputs.size(0)

        targets = []
        for img_in_batch in range(0, batch_size):
            _, boxes, labels = loader.dataset[i + img_in_batch]

            batch_targets = torch.empty(boxes.size(0), 5)
            batch_targets[:, :4] = boxes
            batch_targets[:, 4] = labels
            targets.append(batch_targets)

        with torch.no_grad():
            pred_boxes, pred_labels = model(inputs)
            loc_loss, cls_loss = loss_fn(pred_boxes, target_boxes, pred_labels, target_labels)
            loss = loc_loss + cls_loss

        loss_avg.update(loss.item())

        pred_labels = pred_labels.float()
        dets = encoder.decode(pred_boxes, pred_labels)

        for det, target in zip(dets, targets):
            metric_fn.update_value(det, target)

        status = "{0}[Test][{1}] Loss_avg: {2:.5}, Loss: {3:.5}, Loc_loss: {4:.5}, Cls_loss: {5:.5}".format(
            prefix, i, loss_avg.avg, loss_avg.val, loc_loss, cls_loss
        )

    metric_fn.calculate_value()
    if get_key_metric is not None:
        status = status + ", Metric_avg: {0:.5}".format(get_key_metric(metric_fn.get_metric_value()))
    iterator.set_description(status)
    output = {"metric": metric_fn.get_metric_value(), "loss": loss_avg.avg}
    return output


def set_description_hook_semseg(iterator, epoch, output_train, output_test):
    """ Default epoch set_description hook for semantic segmentation.

    Arguments:
        iterator (iter): iterator.
        epoch (int): number of epoch to store.
        output_train (dict): description of the train stage.
        output_test (dict): description of the test stage.
    """
    if hasattr(iterator, "set_description"):
        print(output_test)
        iterator.set_description(
            "epoch: {0}, test_miou: {1:.5}, train_loss: {2:.5}, test_loss: {3:.5}".format(
                epoch, output_test["metric"][0], output_train["loss"], output_test["loss"]
            )
        )
    if hasattr(iterator, "set_extra_description"):
        iterator.set_extra_description("test_ious", output_test["metric"][1])


def set_description_hook_classification(iterator, epoch, output_train, output_test):
    """ TODO: Write description.
    Default epoch set_description hook for semantic segmentation.

    Arguments:
        iterator (iter): iterator.
        epoch (int): number of epoch to store.
        output_train (dict): description of the train stage.
        output_test (dict): description of the test stage.
    """
    if hasattr(iterator, "set_description"):
        iterator.set_description(
            "epoch: {0}, test_accuracy: {1:.5}, train_loss: {2:.5}, test_loss: {3:.5}".format(
                epoch, output_test["metric"]["accuracy"], output_train["loss"], output_test["loss"]
            )
        )


def end_epoch_hook_semseg(iterator, epoch, output_train, output_test):
    """ Default end_epoch_hook hook for semantic segmentation.
    Arguments:
        iterator (iter): iterator.
        epoch (int): number of epoch to store.
        output_train (dict): description of the train stage.
        output_test (dict): description of the test stage.
        trainer (Trainer): trainer object.
    """
    if hasattr(iterator, "set_description"):
        iterator.set_description(
            "epoch: {0}, test_miou: {1:.5}, train_loss: {2:.5}, test_loss: {3:.5}".format(
                epoch, output_test["metric"]["mean_iou"], output_train["loss"], output_test["loss"]
            )
        )

    if hasattr(iterator, "set_extra_description"):
        iterator.set_extra_description("test_ious", output_test["metric"]["iou"])


def end_epoch_hook_classification(iterator, epoch, output_train, output_test):
    """ Default end_epoch_hook for classification tasks.
    Arguments:
        iterator (iter): iterator.
        epoch (int): number of epoch to store.
        output_train (dict): description of the train stage.
        output_test (dict): description of the test stage.
        trainer (Trainer): trainer object.
    """
    if hasattr(iterator, "set_description"):
        iterator.set_description(
            "epoch: {0}, test_top1: {1:.5}, train_loss: {2:.5}, test_loss: {3:.5}".format(
                epoch, output_test["metric"]["top1"], output_train["loss"], output_test["loss"]
            )
        )


def end_epoch_hook_detection(iterator, epoch, output_train, output_test):
    """ Default end_epoch_hook for detection tasks.
    Arguments:
        iterator (iter): iterator.
        epoch (int): number of epoch to store.
        output_train (dict): description of the train stage.
        output_test (dict): description of the test stage.
        trainer (Trainer): trainer object.
    """
    if hasattr(iterator, "set_description"):
        iterator.set_description(
            "epoch: {0}, test_AP: {1:.5}, train_loss: {2:.5}, test_loss: {3:.5}".format(
                epoch, output_test["metric"]["mAP"], output_train["loss"], output_test["loss"]
            )
        )


class IteratorWithStorage(tqdm):
    """ Class to store logs of deep learning experiments."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = None

    def init_metrics(self, keys=None):
        """ Initialise metrics list.

        Arguments:
            keys (list): list of available keys that need to store.
        """
        if keys is None:
            keys = []
        self.metrics = {k: [] for k in keys}

    def get_metrics(self):
        """ Get stored metrics.

        Returns:
            Dictionary of stored metrics.
        """
        return self.metrics

    def reset_metrics(self):
        """ Reset stored metrics. """
        for key, _ in self.metrics.items():
            self.metrics[key] = []

    def set_description(self, desc=None, refresh=True):
        """ Set description which will be view in status bar.

        Arguments:
            desc (str, optional): description of the iteration.
            refresh (bool): refresh description.
        """
        self.desc = desc or ''
        if self.metrics is not None:
            self._store_metrics(desc)
        if refresh:
            self.refresh()

    def set_extra_description(self, key, val):
        """ Set extra description which will not be view in status bar.

        Arguments:
            key (str): key of the extra description.
            val (str): value of the extra description.
        """
        if self.metrics is not None and key in self.metrics:
            self.metrics[key] = val

    def _store_metrics(self, format_string):
        metrics = dict(x.split(": ") for x in format_string.split(", "))
        for key, val in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(float(val))
