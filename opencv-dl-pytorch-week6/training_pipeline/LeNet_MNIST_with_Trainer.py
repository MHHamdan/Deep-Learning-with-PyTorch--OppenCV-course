# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # <font style="color:blue">Combine them all: LeNet5 pipeline with Trainer</font>
#
# Let's take a look at how we can build the training pipeline using the Trainer helper class and the other helper classes we've discussed before in this notebook.
# Import all the necessary classes and functions:

# %%
# %matplotlib notebook
# %load_ext autoreload
# %autoreload 2

from operator import itemgetter

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import MultiStepLR

from trainer import Trainer, hooks, configuration
from trainer.utils import setup_system, patch_configs
from trainer.metrics import AccuracyEstimator
from trainer.tensorboard_visualizer import TensorBoardVisualizer

# %% [markdown]
# ## <font style="color:Green">1. Get Training and Validation Data Loader</font>
#
#
# Define the data wrappers and transformations (the same way as before):


# %%
def get_data(batch_size, data_root='data', num_workers=1):

    train_test_transforms = transforms.Compose([
        # Resize to 32X32
        transforms.Resize((32, 32)),
        # this re-scales image tensor values between 0-1. image_tensor /= 255
        transforms.ToTensor(),
        # subtract mean (0.1307) and divide by variance (0.3081).
        # This mean and variance is calculated on training data (verify yourself)
        transforms.Normalize((0.1307, ), (0.3081, ))
    ])

    # train dataloader
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=data_root, train=True, download=True, transform=train_test_transforms),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # test dataloader
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=data_root, train=False, download=True, transform=train_test_transforms),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return train_loader, test_loader


# %% [markdown]
# ## <font style="color:Green">2. Define the Model</font>
#
# Define the model (the same way as before):


# %%
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        self._body = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self._head = nn.Sequential(
            nn.Linear(in_features=16 * 5 * 5, out_features=120), nn.ReLU(inplace=True),
            nn.Linear(in_features=120, out_features=84), nn.ReLU(inplace=True),
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, x):
        x = self._body(x)
        x = x.view(x.size()[0], -1)
        x = self._head(x)
        return x


# %% [markdown]
# ## <font style="color:Green">3. Start Experiment / Training</font>
#
#
# Define the experiment with the given model and given data. It's the same idea again: we keep the less-likely-to-change things inside the object and configure it with the things that are more likely to change.
#
# You may wonder, why do we put the specific metric and optimizer into the experiment code and not specify them as parameters. But the experiment class is just a handy way to store all the parts of your experiment in one place. If you change the loss function, or the optimizer, or the model - it seems like another experiment, right? So it deserves to be a separate class.
#
# The Trainer class inner structure is a bit more complicated compared to what we've discussed above - it is just to be able to cope with the different kinds of the tasks we will discuss in this course. We will elaborate a bit more on the Trainer inner structure in the following lectures and now take a look at how compact and self-descriptive the code is:


# %%
class Experiment:
    def __init__(
        self,
        system_config: configuration.SystemConfig = configuration.SystemConfig(),
        dataset_config: configuration.DatasetConfig = configuration.DatasetConfig(),
        dataloader_config: configuration.DataloaderConfig = configuration.DataloaderConfig(),
        optimizer_config: configuration.OptimizerConfig = configuration.OptimizerConfig()
    ):
        self.loader_train, self.loader_test = get_data(
            batch_size=dataloader_config.batch_size,
            num_workers=dataloader_config.num_workers,
            data_root=dataset_config.root_dir
        )
        
        setup_system(system_config)

        self.model = LeNet5()
        self.loss_fn = nn.CrossEntropyLoss()
        self.metric_fn = AccuracyEstimator(topk=(1, ))
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay,
            momentum=optimizer_config.momentum
        )
        self.lr_scheduler = MultiStepLR(
            self.optimizer, milestones=optimizer_config.lr_step_milestones, gamma=optimizer_config.lr_gamma
        )
        self.visualizer = TensorBoardVisualizer()

    def run(self, trainer_config: configuration.TrainerConfig) -> dict:

        device = torch.device(trainer_config.device)
        self.model = self.model.to(device)
        self.loss_fn = self.loss_fn.to(device)

        model_trainer = Trainer(
            model=self.model,
            loader_train=self.loader_train,
            loader_test=self.loader_test,
            loss_fn=self.loss_fn,
            metric_fn=self.metric_fn,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            device=device,
            data_getter=itemgetter(0),
            target_getter=itemgetter(1),
            stage_progress=trainer_config.progress_bar,
            get_key_metric=itemgetter("top1"),
            visualizer=self.visualizer,
            model_saving_frequency=trainer_config.model_saving_frequency,
            save_dir=trainer_config.model_dir
        )

        model_trainer.register_hook("end_epoch", hooks.end_epoch_hook_classification)
        self.metrics = model_trainer.fit(trainer_config.epoch_num)
        return self.metrics


# %%
def main():
    '''Run the experiment
    '''
    # patch configs depending on cuda availability
    dataloader_config, trainer_config = patch_configs(epoch_num_to_set=15)
    dataset_config = configuration.DatasetConfig(root_dir="data")
    experiment = Experiment(dataset_config=dataset_config, dataloader_config=dataloader_config)
    results = experiment.run(trainer_config)

    return results


# %%
if __name__ == '__main__':
    main()

# %% [markdown]
# So in a few lines of code, we got a more robust system than we had before - we have richer visualizations, a more configurable training process, and we separated the pipeline for the training from the model - so we can concentrate on the things that matter the most.

# %% [markdown]
# # <font style="color:blue">References</font>
#
# You may wonder whether it is a common way of doing deep learning or we're overengineering here. We may assure you that this is a common way to do deep learning research in an industry - most of the companies and research groups invest in building these DL training frameworks for their projects, and some of them are even published to the open-source. To name a couple of them:
# - https://github.com/NVlabs/SPADE
# - https://github.com/pytorch/ignite
# - https://github.com/PyTorchLightning/pytorch-lightning
# - https://github.com/catalyst-team/catalyst
# - https://github.com/open-mmlab/mmdetection
# - https://github.com/fastai/fastai

# %%
