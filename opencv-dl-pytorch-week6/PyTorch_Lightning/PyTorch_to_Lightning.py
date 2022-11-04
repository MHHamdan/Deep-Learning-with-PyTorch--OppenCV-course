"""
Usage:
 
    # View available command-line arguments
    python PyTorch_to_Lightning.py -h
    python PyTorch_to_Lightning.py --help
    
    # Usage of Trainer.from_argparse_args(...)
    # Pass keyword arguments of Trainer module directly from the command-line
    # along with other necessary arguments
    
    python PyTorch_to_Lightning.py --max_epochs 10 --gpus 1 --deterministic True  
    # OR
    python PyTorch_to_Lightning.py --max_epochs 10 --gpus 1 --deterministic True  --learning_rate 0.001 --batch_size 256 

    # Demo output
    Global seed set to 21
    GPU available: True, used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

    | Name       | Type         | Params
    --------------------------------------------
    0 | _body      | Sequential   | 2.6 K
    1 | _head      | Sequential   | 59.1 K
    2 | train_acc  | Accuracy     | 0
    3 | valid_acc  | Accuracy     | 0
    4 | train_loss | AverageMeter | 0
    5 | valid_loss | AverageMeter | 0
    --------------------------------------------
    61.7 K    Trainable params
    0         Non-trainable params
    61.7 K    Total params
    0.247     Total estimated model params size (MB)
    Global seed set to 21

    Validation sanity check: 0it [00:00, ?it/s]
    Validation sanity check:   100%|##########| 2/2 [00:00<?, ?it/s]
                                                                

    Training: -1it [00:00, ?it/s]
    Training:   0%|          | 0/138 [00:00<?, ?it/s]
    Epoch 0:  87%|########6 | 120/138 [00:06<00:00, 18.66it/s, loss=2.31, v_num=9, train/batch_acc=0.0742]

    Validating: 0it [00:00, ?it/s][A

    Validating:   0%|          | 0/20 [00:00<?, ?it/s][A

    Validating: 100%|##########| 20/20 [00:04<00:00,  4.87it/s][A
    Epoch 0: 100%|##########| 138/138 [00:11<00:00, 12.46it/s, loss=2.31, v_num=9, train/batch_acc=0.115, valid/acc=0.0876, valid/loss=2.310]
    The latest model path: lightning_logs\version_9\checkpoints\epoch=0-step=117.ckpt
  
"""

# Let’s graduate from  Implementing LeNet in PyTorch notebook to reimplementing it here in PyTorch Lightning.
# Our focus is mostly on implementation, so we shall cover the following:

# 1. Model in PyTorch Lightning (`LightningModule`)
# 2. Training step
# 3. Validation step
# 4. Metrics
# 5. Optimizer configuration
# 6. Data Module (`LightningDataModule`)
# 7. Hyperparameters
# 8. Tensorboard logs
# 9. Saving and loading the model
# 10. Trainer

# For in-depth details open the notebook (.ipynb) version of this code.
# Let's see how the PyTorch code is re-factored to PyTorch Lightning.
# IMPORTS
import os
import types
import random
import inspect
import warnings
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# convert image tensor to PIL image and in range 0-255
from torchvision.transforms.functional import to_pil_image

import pytorch_lightning as pl
from pytorch_lightning import seed_everything  # for reproducibility

# Importing EarlyStopping callback
from pytorch_lightning.callbacks import EarlyStopping

# using Functional API
from torchmetrics.functional import accuracy

# Use Module API to track and compute metrics automatically
from torchmetrics import Accuracy
from torchmetrics import AverageMeter

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

warnings.filterwarnings("ignore", category=UserWarning)  # filter UserWarning


# LIGHTNING MODULE
# Create a class, with LightningModule as the base class.
# This class should have the model, training step, validation step,
# optimizer etc.


class LeNet5(pl.LightningModule):  # here nn.Module is replaced by LightningModule
    def __init__(self, learning_rate=0.01, num_classes=10):
        super().__init__()

        # Save the arguments as hyperparameters.
        self.save_hyperparameters()

        self._body = nn.Sequential(
            # First convolution Layer
            # input size = (32, 32), output size = (28, 28)
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            # ReLU activation
            nn.ReLU(inplace=True),
            # Max pool 2-d
            nn.MaxPool2d(kernel_size=2),
            # Second convolution layer
            # input size = (14, 14), output size = (10, 10)
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # output size = (5, 5)
        )

        # Fully connected layers
        self._head = nn.Sequential(
            # First fully connected layer
            # in_features = total number of weights in last conv layer = 16 * 5 * 5
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            # ReLU activation
            nn.ReLU(inplace=True),
            # second fully connected layer
            # in_features = output of last linear layer = 120
            nn.Linear(in_features=120, out_features=84),
            # ReLU activation
            nn.ReLU(inplace=True),
            # Third fully connected layer. It is also the output layer
            # in_features = output of last linear layer = 84
            # and out_features = number of classes = 10 (MNIST data 0-9)
            nn.Linear(in_features=84, out_features=self.hparams.num_classes),
        )

        # declaring all metrics as attributes of model
        # so they are detected as children

        # in this notebook we'll be using torchmetrics Module metrics
        # initialize Accuracy Module here
        acc_obj = Accuracy(num_classes=self.hparams.num_classes)

        # use .clone() so each metric has its maintain own state
        self.train_acc = acc_obj.clone()
        self.valid_acc = acc_obj.clone()

        # Using AverageMeter Class to accumulate batch loss values
        # and to automate mean calculation
        average_meter = AverageMeter()
        self.train_loss = average_meter.clone()
        self.valid_loss = average_meter.clone()

    def forward(self, x):
        # apply feature extractor
        x = self._body(x)
        # flatten the output of conv layers
        # dimension should be batch_size * number_of weights_in_last conv_layer
        x = x.view(x.size()[0], -1)
        # apply classification head
        x = self._head(x)
        return x

    def on_train_epoch_start(self):
        super().on_train_epoch_start()

        # Reset state variables for train metrics to
        # their default values before start of each epoch

        self.train_acc.reset()
        self.train_loss.reset()

        # self.check_metric_value("Train_acc", self.train_acc)
        # self.check_metric_value("Train_loss", self.train_loss)

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()

        # Reset state variables for validation metrics to
        # their default values before start of each epoch

        self.valid_acc.reset()
        self.valid_loss.reset()

        # self.check_metric_value("Valid_acc", self.valid_acc)
        # self.check_metric_value("Valid_loss", self.valid_loss)

    def training_step(self, batch, batch_idx):

        # get data and labels from batch
        data, target = batch

        # get prediction
        output = self(data)

        # calculate batch loss
        loss = F.cross_entropy(output, target)

        # get probability score using softmax
        prob = F.softmax(output, dim=1)

        # get the index of the max probability
        pred = prob.data.max(dim=1)[1]

        # Using Functional API
        # # -----------------
        # acc = accuracy(pred, target)
        # # -----------------

        # Using Module API
        # # -----------------
        # calculate and accumulate batch accuracy
        acc = self.train_acc(pred, target)

        # accumulate batch loss
        self.train_loss(loss)
        # # -----------------

        # LOG METRICS to a logger. Default: Tensorboard
        # pytorch-lightning automatically adds a running mean of
        # training step loss using the value returned by this method to the progress bar
        # So we can set the prog_bar=False
        # If prog_bar=True, we'll see different values for loss and train/batch_loss in the progress bar
        # This is due to one being a running mean and one being the current batch loss

        # Arguments such as on_epoch, on_step and logger are set automatically depending on
        # hook methods it's been called from
        self.log("train/batch_loss", loss, prog_bar=False)

        # logging and adding current batch_acc to progress_bar
        self.log("train/batch_acc", acc, prog_bar=True)

        # Using Functional API
        # # -----------------
        # we need to return loss and accuracy
        # dic = {
        #     'train_loss': loss.clone().detach().cpu(),
        #     'train_acc': acc.detach().cpu(),
        # }

        # return {'loss': loss, 'batch_metrics': dic}
        # # -----------------

        # Using Module API, we only need to return the loss
        # # -----------------
        return loss
        # # -----------------

    def training_epoch_end(self, training_step_outputs):
        """Calculate epoch level metrics for the train set"""
        # Using Functional API
        # # -----------------
        # # training_step_outputs = [{'loss': loss, 'batch_metrics': dic}, {...}, {...}]
        # avg_train_loss = torch.tensor([x['batch_metrics']['train_loss'] for x in training_step_outputs]).mean()
        # avg_train_acc = torch.tensor([x['batch_metrics']['train_acc'] for x in training_step_outputs]).mean()
        # # -----------------

        # Using Module API
        # # -----------------
        # # Compute epoch loss and accuracy
        avg_train_loss = self.train_loss.compute()
        avg_train_acc = self.train_acc.compute()
        # # -----------------

        self.log("train/loss", avg_train_loss, prog_bar=True)
        self.log("train/acc", avg_train_acc, prog_bar=True)
        # Set X-axis as epoch number for epoch-level metrics
        self.log("step", self.current_epoch)

    def validation_step(self, batch, batch_idx):

        # get data and labels from batch
        data, target = batch

        # get prediction
        output = self(data)

        # calculate loss
        loss = F.cross_entropy(output, target)

        # get probability score using softmax
        prob = F.softmax(output, dim=1)

        # get the index of the max probability
        pred = torch.argmax(prob, dim=1)

        # Using Functional API
        # # -----------------
        # acc = accuracy(pred, target)

        # dic = {
        #     'v_loss': loss.cpu(),
        #     'v_acc': acc.cpu(),
        # }

        # return dic
        # # -----------------

        # Using Module API
        # # -----------------
        # accumulate validation accuracy and loss
        self.valid_acc(pred, target)
        self.valid_loss(loss)
        # # -----------------

        # we don't log validation step values as we are
        # more interested in epoch-level values for the
        # validation step

    def validation_epoch_end(self, validation_step_outputs):
        """Calculate epoch level metrics for the validation set"""
        # Using Functional API
        # # -----------------
        # # validation_step_outputs = [dic, ..., dic]

        # avg_val_loss = torch.tensor([x['v_loss'] for x in validation_step_outputs]).mean()
        # avg_val_acc = torch.tensor([x['v_acc'] for x in validation_step_outputs]).mean()
        # # -----------------

        # Using Module API
        # # -----------------
        avg_val_loss = self.valid_loss.compute()
        avg_val_acc = self.valid_acc.compute()
        # # -----------------

        self.log("valid/loss", avg_val_loss, prog_bar=True)
        self.log("valid/acc", avg_val_acc, prog_bar=True)

        # use epoch as X-axis
        self.log("step", self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)

    def check_metric_value(self, name, metric_attribute):
        """
        To check if the training and validation metrics
        state variables are reset
        """
        to_check = ("tp", "fp", "tn", "fn", "correct", "total", "value", "weight")
        print(f"Epoch:{self.current_epoch}. Name:{name}")
        for i in inspect.getmembers(metric_attribute):

            # to remove private and protected functions
            if not i[0].startswith("_") and i[0] in to_check:
                # To remove other methods that
                # doesnot start with a underscore
                # if not inspect.ismethod(i[1]) and not isinstance(i[1], types.FunctionType):
                print(i)
        print("----------")


# LIGHTNING DATA MODULE
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_root, batch_size, num_workers):

        super().__init__()

        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_test_transforms = transforms.Compose(
            [
                # Resize to 32X32
                transforms.Resize((32, 32)),
                # this re-scales image tensor values between 0-1. image_tensor /= 255
                transforms.ToTensor(),
                # subtract mean (0.1307) and divide by variance (0.3081).
                # This mean and variance are calculated on training data (verify for yourself)
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def prepare_data(self):
        # download
        datasets.MNIST(self.data_root, train=True, download=True)
        datasets.MNIST(self.data_root, train=False, download=True)

    def setup(self, stage=None):
        # create splits
        self.train_dataset = datasets.MNIST(
            self.data_root, train=True, transform=self.train_test_transforms
        )
        self.val_dataset = datasets.MNIST(
            self.data_root, train=False, transform=self.train_test_transforms
        )

    def train_dataloader(self):
        # train loader
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return train_loader

    def val_dataloader(self):
        # validation loader
        test_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return test_loader


def configuration_parser(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    # additional arguments
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    return parser


def training_validation():

    # random seed
    seed_everything(21)

    # initiate the argument parser
    parser = ArgumentParser()

    # initiates Trainer and gets already added arguments in the Trainer
    parser = pl.Trainer.add_argparse_args(parser)

    # adds more arguments to the argument parser
    parser = configuration_parser(parser)

    # using "parse_known_args" to work argument parser in the notebook.
    # For normal python file we can use "parse_args"
    # args = parser.parse_args()
    args = parser.parse_args()

    # init model
    model = LeNet5(learning_rate=args.learning_rate)

    # init the data module
    data_module = MNISTDataModule(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # To pass arguments directly form command line, initiate the trainer like below
    # We'll pass argumnets such as gpus, max_epochs, determinstic directly from the command line
    # along with additional arguments such as batch_size and learning rate
    trainer = pl.Trainer.from_argparse_args(
        args,
        #  fast_dev_run=True,  # use to test the code pipeline
        # max_epochs=3,  # maximum number of epoch
        # deterministic=True,  # to make code reproducible
        # gpus=1,  # total number of GPUs
        progress_bar_refresh_rate=20,  # (steps) how often to update the progress bar values
        log_every_n_steps=20,  # (steps) how often we want to write the training_step and validation_step metrics to a logger
        callbacks=[
            EarlyStopping(
                monitor="valid/loss"
            )  # stop training if validation loss does not decrease for 3 epochs (default)
        ],
    )
    # start training
    trainer.fit(model, data_module)
    return model, data_module


def denormalize(tensors):
    """Denormalizes image tensors back to range [0.0, 1.0]"""

    mean = torch.as_tensor((0.1307,))
    std = torch.as_tensor((0.3081,))

    tensors = tensors.clone()
    tensors[:, 0, :, :].mul_(std[0]).add_(mean[0])

    return torch.clamp(tensors.cpu(), 0.0, 1.0)


def sample_prediction(model, data_module, ckpt_path):
    random.seed(21)

    # load model from checkpoint
    ckpt_model = model.load_from_checkpoint(ckpt_path)
    # freeze model for inference
    ckpt_model.freeze()

    # run model in evaluation mode
    ckpt_model.eval()

    # get val_dataloader for data_module
    val_data = data_module.val_dataloader()

    imgs = []
    preds = []
    probs = []

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ckpt_model.to(device)

    for data, _ in val_data:

        with torch.no_grad():
            output = ckpt_model(data.to(device))

        # get probability score using softmax
        prob = F.softmax(output, dim=1)

        # get the max probability
        pred_prob = prob.data.max(dim=1)[0]

        # get the index of the max probability
        pred_index = prob.data.max(dim=1)[1]
        # pass the loaded model
        pred = pred_index.cpu().tolist()
        prob = pred_prob.cpu().tolist()

        imgs.extend([np.asarray(to_pil_image(image)) for image in denormalize(data)])

        preds.extend(pred)
        probs.extend(prob)
        break

    # randomly select 15 images fom the first batch
    random_15 = random.sample(list(zip(imgs, preds, probs)), 15)

    plt.figure(figsize=(9, 10))

    for idx, (img, pred, prob) in enumerate(random_15, 1):
        img = np.array(img).reshape(32, 32)
        plt.subplot(5, 3, idx)
        plt.imshow(img, cmap="gray")
        plt.title(f"Prediction: {pred} Prob: {prob:.2f}")
        plt.axis("off")

    plt.show()

    return


def get_latest_run_version_ckpt_epoch_no(
    lightning_logs_dir="lightning_logs", run_version=None
):
    if run_version is None:
        run_version = 0
        for dir_name in os.listdir(lightning_logs_dir):
            if "version" in dir_name:
                if int(dir_name.split("_")[1]) > run_version:
                    run_version = int(dir_name.split("_")[1])

    checkpoints_dir = os.path.join(
        lightning_logs_dir, "version_{}".format(run_version), "checkpoints"
    )

    files = os.listdir(checkpoints_dir)
    ckpt_filename = None
    for file in files:
        if file.endswith(".ckpt"):
            ckpt_filename = file

    if ckpt_filename is not None:
        ckpt_path = os.path.join(checkpoints_dir, ckpt_filename)
    else:
        print("CKPT file is not present")

    return ckpt_path


# Let's start training.

if __name__ == "__main__":
    # view logs
    # tensorboard --logdir=lightning_logs

    model, data_module = training_validation()

    # get checkpoints of the latest run
    ckpt_path = get_latest_run_version_ckpt_epoch_no()
    print("The latest model path: {ckpt_path}")

    # sample prediction
    sample_prediction(model, data_module, ckpt_path)

# References
# 1. https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# 1. https://pytorch.org/tutorials/beginner/saving_loading_models.html
# 1. https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=gEulmrbxwaYL
# 1. https://pytorch-lightning.readthedocs.io
# 1. https://github.com/PyTorchLightning/pytorch-lightning
# 1. https://www.youtube.com/watch?v=QHww1JH7IDU
# 1. https://pytorch-lightning.readthedocs.io/en/latest/
# 1. https://www.youtube.com/channel/UC8m-y0yAFJpX0hRvxH8wJVw/featured
