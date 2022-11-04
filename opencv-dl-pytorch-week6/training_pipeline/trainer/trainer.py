# # <font style="color:blue">Trainer Class</font>
#
# **This is a generic class for training loop.**
#
# Trainer class is equivalent to the `main` method. 
#
# In the main method, we were passing configurations, the model, optimizer, learning rate scheduler, and the number of epochs.  It was calling the method to get the train and test data loader. Using these, it is training and validating the model. During training and validation, it was also sending logs to TensorBoard and saving the model.
#
# The trainer class is doing the same in a more modular way so that we can experiment with different loss functions, different visualizers, different types of targets, etc. 
#

"""Unified class to make training pipeline for deep neural networks."""
import os
import datetime

from typing import Union, Callable
from pathlib import Path
from operator import itemgetter

import torch

from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .hooks import test_hook_default, train_hook_default
from .visualizer import Visualizer


# ## <font style="color:blue">Trainer Class Methods and its Parameters</font>
#
# ### <font style="color:green">  \_\_init\_\_ </font>
#
# Setting different attributes.
#
# **Parameters:**
#
# - `model` : `nn.Module` - torch model to train
#
#         
# - `loader_train` : `torch.utils.DataLoader` - train dataset loader.
#
#     
# - `loader_test` : `torch.utils.DataLoader` - test dataset loader
#
#        
# - `loss_fn` : `callable` - loss function. In the main function, the cross-entropy loss was being used; here, we can pass the loss we want to use. For example, if we are solving a regression problem, we can not use cross-entropy loss. It is better to use RMS-loss.
#
#
#         
# - `metric_fn` : `callable` - evaluation metric function. In the main function, we had loss and accuracy as our evaluation metric. Here we can pass any evaluation metric. For example, in a detection problem, we need a precision-recall metric instead of accuracy.
#
#         
# - `optimizer` : `torch.optim.Optimizer` - Optimizer.
#
#         
# - `lr_scheduler` : `torch.optim.LrScheduler` - Learning Rate scheduler.
#
#         
# - `configuration` : `TrainerConfiguration` - a set of training process parameters.
#
# Here, we need a data iterator and target iterator separately, because we are writing a general trainer class. For example, for the detection problem for a single image, we might have `n`-number of objects and their coordinates. 
#
#         
# - `data_getter` : `Callable` - function object to extract input data from the sample prepared by dataloader.
#
#         
# - `target_getter` : `Callable` - function object to extract target data from the sample prepared by dataloader.
#
#         
# - `visualizer` : `Visualizer` - optional, shows metrics values (various backends are possible). We can pass the visualizer of our choice. For example, Matplotlib based visualizer, TensorBoard based, etc.
#
# It is also calling its method `_register_default_hooks` what this method does we will see next. In short, this is making sure that training and validation function is registered at the time of trainer class object initiation. 
#
#
# ### <font style="color:green"> _register_default_hooks </font>
#
# It is calling the another method `register_hook` to register training (`train_hook_default`) and validation (`test_hook_default`) functions. `train_hook_default` and `test_hook_default` are defined in the `hook`-module.  We will go in details in the module.
#
#
# ### <font style="color:green"> register_hook </font>
#
# It is updating the key-value pair of a dictionary, where the key is string and value is a callable function.
#
# **Parameters:**
#
# - `hook_type`: `string` - hook type. For example, wether the function will be used for train or test.
#
#
# - `hook_fn`: `callable` - hook function.
#
#
#
#
# ### <font style="color:green"> fit </font>
#
# Taking the number of epochs and training and validating the model. It is also adding logs to the visualizer. 
#
# **Parameters:**
#
# - `epochs`: `int` - number of epochs to train model.
#

class Trainer:  # pylint: disable=too-many-instance-attributes
    """ Generic class for training loop.

    Parameters
    ----------
    model : nn.Module
        torch model to train
    loader_train : torch.utils.DataLoader
        train dataset loader.
    loader_test : torch.utils.DataLoader
        test dataset loader
    loss_fn : callable
        loss function
    metric_fn : callable
        evaluation metric function
    optimizer : torch.optim.Optimizer
        Optimizer
    lr_scheduler : torch.optim.LrScheduler
        Learning Rate scheduler
    configuration : TrainerConfiguration
        a set of training process parameters
    data_getter : Callable
        function object to extract input data from the sample prepared by dataloader.
    target_getter : Callable
        function object to extract target data from the sample prepared by dataloader.
    visualizer : Visualizer, optional
        shows metrics values (various backends are possible)
    # """
    def __init__( # pylint: disable=too-many-arguments
        self,
        model: torch.nn.Module,
        loader_train: torch.utils.data.DataLoader,
        loader_test: torch.utils.data.DataLoader,
        loss_fn: Callable,
        metric_fn: Callable,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Callable,
        device: Union[torch.device, str] = "cuda",
        model_saving_frequency: int = 1,
        save_dir: Union[str, Path] = "checkpoints",
        model_name_prefix: str = "model",
        data_getter: Callable = itemgetter("image"),
        target_getter: Callable = itemgetter("target"),
        stage_progress: bool = True,
        visualizer: Union[Visualizer, None] = None,
        get_key_metric: Callable = itemgetter("top1"),
    ):
        self.model = model
        self.loader_train = loader_train
        self.loader_test = loader_test
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.model_saving_frequency = model_saving_frequency
        self.save_dir = save_dir
        self.model_name_prefix = model_name_prefix
        self.stage_progress = stage_progress
        self.data_getter = data_getter
        self.target_getter = target_getter
        self.hooks = {}
        self.visualizer = visualizer
        self.get_key_metric = get_key_metric
        self.metrics = {"epoch": [], "train_loss": [], "test_loss": [], "test_metric": []}
        self._register_default_hooks()

    def fit(self, epochs):
        """ Fit model method.

        Arguments:
            epochs (int): number of epochs to train model.
        """
        iterator = tqdm(range(epochs), dynamic_ncols=True)
        for epoch in iterator:
            output_train = self.hooks["train"](
                self.model,
                self.loader_train,
                self.loss_fn,
                self.optimizer,
                self.device,
                prefix="[{}/{}]".format(epoch, epochs),
                stage_progress=self.stage_progress,
                data_getter=self.data_getter,
                target_getter=self.target_getter
            )
            output_test = self.hooks["test"](
                self.model,
                self.loader_test,
                self.loss_fn,
                self.metric_fn,
                self.device,
                prefix="[{}/{}]".format(epoch, epochs),
                stage_progress=self.stage_progress,
                data_getter=self.data_getter,
                target_getter=self.target_getter,
                get_key_metric=self.get_key_metric
            )
            if self.visualizer:
                self.visualizer.update_charts(
                    None, output_train['loss'], output_test['metric'], output_test['loss'],
                    self.optimizer.param_groups[0]['lr'], epoch
                )

            self.metrics['epoch'].append(epoch)
            self.metrics['train_loss'].append(output_train['loss'])
            self.metrics['test_loss'].append(output_test['loss'])
            self.metrics['test_metric'].append(output_test['metric'])

            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    self.lr_scheduler.step(output_train['loss'])
                else:
                    self.lr_scheduler.step()

            if self.hooks["end_epoch"] is not None:
                self.hooks["end_epoch"](iterator, epoch, output_train, output_test)

            if (epoch + 1) % self.model_saving_frequency == 0:
                os.makedirs(self.save_dir, exist_ok=True)
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.save_dir, self.model_name_prefix) + str(datetime.datetime.now())
                )
        return self.metrics

    def register_hook(self, hook_type, hook_fn):
        """ Register hook method.

        Arguments:
            hook_type (string): hook type.
            hook_fn (callable): hook function.
        """
        self.hooks[hook_type] = hook_fn

    def _register_default_hooks(self):
        self.register_hook("train", train_hook_default)
        self.register_hook("test", test_hook_default)
        self.register_hook("end_epoch", None)
