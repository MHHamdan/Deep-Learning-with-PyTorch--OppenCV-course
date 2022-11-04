"""Implementation of several hooks that used in a Trainer class."""
from operator import itemgetter

import torch

from tqdm.auto import tqdm

from .utils import AverageMeter


def train_hook_faster_rcnn(
    model,
    loader,
    optimizer,
    device,
    data_getter=None,  
    target_getter=None, 
    iterator_type=tqdm,
    prefix="",
    stage_progress=False
):
    """ Default train loop function.

    Arguments:
        model (nn.Module): torch model which will be train.
        loader (torch.utils.DataLoader): dataset loader.
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
        
        images = list(image.to(device) for image in sample[0])
        targets = [{key: value.to(device) for key, value in target.items()} for target in sample[1]]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        
        losses.backward()

        optimizer.step()
        loss_avg.update(losses.item())
        status = "{0}[Train][{1}] Loss_avg: {2:.5}, Loss: {3:.5}, loss_box_reg: {4:.5}, loss_classifier: {5:.5}, loss_objectness: {6:.5}, loss_rpn_box_reg: {7:.5}, LR: {8:.5}".format(
            prefix, i, loss_avg.avg, loss_avg.val, loss_dict['loss_box_reg'].item(), loss_dict['loss_classifier'].item(), loss_dict['loss_objectness'].item(), loss_dict['loss_rpn_box_reg'].item(), optimizer.param_groups[0]['lr']
        )
        iterator.set_description(status)
    return {"loss": loss_avg.avg}


def test_hook_faster_rcnn(
    model,
    loader,
    metric_fn,
    device,
    data_getter=None,  
    target_getter=None,
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
    metric_fn.reset()

    for i, sample in enumerate(iterator):
        
        image = list(image.to(device) for image in sample[0])
        targets = [{key: value.to(device) for key, value in target.items()} for target in sample[1]]
        
        eval_targets = []
        
        for target in targets:
            bboxes = target['boxes']
            labels = target['labels']
            img_targets = torch.empty((len(labels), 5))
            img_targets[:, :4] = bboxes
            img_targets[:, 4] = labels
            eval_targets.append(img_targets)
                   
            
        with torch.no_grad():
            detections = model(image)
            
        predictions = []
        for dets in detections:
            bboxes = dets['boxes']
            labels = dets['labels']
            scores = dets['scores']
            img_det = torch.empty((len(labels), 6))
            img_det[:, :4] = bboxes
            img_det[:, 4] = scores
            img_det[:, 5] = labels
            predictions.append(img_det)
            
        for det, target in zip(predictions, eval_targets):
            metric_fn.update_value(det, target)
            
        
    metric_fn.calculate_value()
    if get_key_metric is not None:
        status = "Metric_avg: {0:.5}".format(get_key_metric(metric_fn.get_metric_value()))
    iterator.set_description(status)
    output = {"metric": metric_fn.get_metric_value()}
    return output


def end_epoch_hook_faster_rcnn(iterator, epoch, output_train, output_test):
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
            "epoch: {0}, test_AP: {1:.5}, train_loss: {2:.5}".format(
                epoch, output_test["metric"]["mAP"], output_train["loss"]
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
