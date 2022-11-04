"""
Usage:
    python Transfer_Learning_with_Lightning.py --gpu_count 1 --max_epochs 5 --deterministic True --batch_size 16
"""

# IMPORTS
import os
import random
import zipfile
import warnings
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# convert image tensor to PIL image and in range 0-255
from torchvision.transforms.functional import to_pil_image

import pytorch_lightning as pl
from pytorch_lightning import seed_everything  # for reproducibility

# Importing EarlyStopping callback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


# Use Module API to calculate and track and compute metric for automatically
from torchmetrics import Accuracy
from torchmetrics import AverageMeter

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from torchvision import datasets, transforms

warnings.filterwarnings("ignore", category=UserWarning)  # filter UserWarning


# We have chosen the ResNet family for transfer learning/fine-tuning.
# - The ResNet family model has five  layers: `layer1`, `layer2`, `layer3`, `layer4`, and `fc`.
# - It is mandatory to replace (and re-train) the last fully connected layer (`fc`) for fine-tuning.
# - How many more layers should be fine-tuned to get the best result is something you will know only
# by practically working on it each time. So, we have written a `LightningModule` class
# that takes `fine_tune_start` as an argument and updates the `requires_grad` parameters  of the
# ResNet model accordingly.


class TransferLearningWithResNet(pl.LightningModule):
    def __init__(
        self,
        resnet_model_name="resnet18",
        pretrained=True,
        fine_tune_start=1,
        learning_rate=0.01,
        num_classes=10,
    ):
        super().__init__()

        self.save_hyperparameters()

        resnet = getattr(models, self.hparams.resnet_model_name)(
            pretrained=self.hparams.pretrained
        )

        if pretrained:
            for param in resnet.parameters():
                param.requires_grad = False

        if pretrained and self.hparams.fine_tune_start <= 1:
            for param in resnet.layer1.parameters():
                param.requires_grad = True

        if pretrained and self.hparams.fine_tune_start <= 2:
            for param in resnet.layer2.parameters():
                param.requires_grad = True

        if pretrained and self.hparams.fine_tune_start <= 3:
            for param in resnet.layer3.parameters():
                param.requires_grad = True

        if pretrained and self.hparams.fine_tune_start <= 4:
            for param in resnet.layer4.parameters():
                param.requires_grad = True

        last_layer_in = resnet.fc.in_features
        resnet.fc = nn.Linear(last_layer_in, self.hparams.num_classes)

        self.resnet = resnet

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

        return self.resnet(x)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()

        # Reset state variables for train metrics to
        # their default values before start of each epoch

        self.train_acc.reset()
        self.train_loss.reset()

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()

        # Reset state variables for validation metrics to
        # their default values before start of each epoch

        self.valid_acc.reset()
        self.valid_loss.reset()

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

        # calculate and accumulate batch accuracy
        acc = self.train_acc(pred, target)

        # accumulate batch loss
        self.train_loss(loss)
        # # -----------------

        # LOG METRICS to a logger. Default: Tensorboard
        self.log("train/batch_loss", loss, prog_bar=False)
        self.log("train/batch_acc", acc, prog_bar=True)

        return loss

    def training_epoch_end(self, training_step_outputs):
        # compute epoch loss and accuracy
        avg_train_loss = self.train_loss.compute()
        avg_train_acc = self.train_acc.compute()

        # log metrics
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

        # accumulate validation accuracy and loss

        self.valid_acc(pred, target)
        self.valid_loss(loss)
        # no need to return anything explicitly

    def validation_epoch_end(self, validation_step_outputs):
        # compute epoch-level metric
        avg_val_loss = self.valid_loss.compute()
        avg_val_acc = self.valid_acc.compute()

        # log metrics
        self.log("valid/loss", avg_val_loss, prog_bar=True)
        self.log("valid/acc", avg_val_acc, prog_bar=True)
        # use epoch as X-axis
        self.log("step", self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)


# We will be using a dataset from kaggle. link: https://www.kaggle.com/ashishsaxena2209/animal-image-datasetdog-cat-and-panda
# It consists of `1000` images each of multiple animals, with all images of a particular animal in a separate folder.
# We have split it into `80:20` ratio for the `train: validation`, which you can download from
# here:https://www.dropbox.com/sh/n5nya3g3airlub6/AACi7vaUjdTA0t2j_iKWgp4Ra?dl=1


# Let's create the Lightning data module


class CatDogPandaDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers):

        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        preprocess = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
        )

        self.common_transforms = transforms.Compose(
            [preprocess, transforms.Normalize(mean, std)]
        )

        self.aug_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(256),
                transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
                transforms.RandomGrayscale(p=0.1),
                self.common_transforms,
                transforms.RandomErasing(),
            ]
        )

    def prepare_data(self):

        curr_dir = os.getcwd()
        print("Preparing Cat, Dog, Panda dataset")
        url = (
            "https://www.dropbox.com/sh/n5nya3g3airlub6/AACi7vaUjdTA0t2j_iKWgp4Ra?dl=1"
        )
        filename = os.path.join(curr_dir, r"animal-data.zip")
        root = curr_dir

        torchvision.datasets.utils.download_url(url, root, filename)

        with zipfile.ZipFile(filename, "r") as f:

            # list the folders present in the zipfile
            directories = [info.filename for info in f.infolist() if info.is_dir()]

            # index 1 contatins the name of the root directory we are interested in
            # ["/", "cat-dog-panda/", "cat-dog-panda/training", ...]
            self.data_root = os.path.join(curr_dir, directories[1])

            # if data has not been extracted already (useful when experimenting again)
            # avoids extracting if dataset already extracted before
            if not os.path.isdir(self.data_root):
                # extract the zipfile contents
                f.extractall(curr_dir)

        print("Preparation completed.")

    def setup(self, stage=None):
        train_data_path = os.path.join(self.data_root, "training")
        val_data_path = os.path.join(self.data_root, "validation")

        self.train_dataset = datasets.ImageFolder(
            root=train_data_path, transform=self.aug_transforms
        )
        self.val_dataset = datasets.ImageFolder(
            root=val_data_path, transform=self.common_transforms
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
    parser.add_argument("--gpu_count", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--resnet_model_name", type=str, default="resnet18")
    parser.add_argument("--pretrained", type=bool, default=True)
    parser.add_argument("--fine_tune_start", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=3)
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

    model = TransferLearningWithResNet(
        resnet_model_name=args.resnet_model_name,
        pretrained=args.pretrained,
        fine_tune_start=args.fine_tune_start,
        num_classes=args.num_classes,
        learning_rate=args.learning_rate,
    )

    data_module = CatDogPandaDataModule(
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="valid/acc",
        mode="max",
        filename="transfer_learning-epoch-{epoch:02d}-valid_acc-{valid/acc:.2f}",
        auto_insert_metric_name=False,
        save_weights_only=False,
    )

    early_stopping_callback = EarlyStopping(monitor="valid/loss")

    # most basic trainer, uses good defaults
    # we'll pass some defaults via command-line arguments
    trainer = pl.Trainer.from_argparse_args(
        args,
        #  fast_dev_run=True,  # use to test the code pipeline
        # max_epochs=3,  # maximum number of epoch
        # deterministic=True,  # to make code reproducible
        gpus=args.gpu_count,  # total number of GPUs
        progress_bar_refresh_rate=20,  # (steps) how often to update the progress bar values
        log_every_n_steps=20,  # (steps) how often we want to write the training_step and validation_step metrics to a logger
        callbacks=[early_stopping_callback, checkpoint_callback],
    )
    # start training
    trainer.fit(model, data_module)
    return model, data_module


def denormalize(tensors):
    """Denormalizes image tensors back to range [0.0, 1.0]"""

    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])

    tensors = tensors.clone()
    for c in range(3):
        tensors[:, c, :, :].mul_(std[c]).add_(mean[c])

    return torch.clamp(tensors.cpu(), 0.0, 1.0)


def sample_prediction(model, data_module, ckpt_path):
    random.seed(1)

    # load model from checkpoint
    ckpt_model = model.load_from_checkpoint(ckpt_path)
    # freeze model for inference
    ckpt_model.freeze()

    # run model in evaluation mode
    ckpt_model.eval()

    # get val_dataloader for data_module
    val_data = data_module.val_dataloader()

    idx_to_class = {j: i for i, j in data_module.val_dataset.class_to_idx.items()}

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

    # randomly select 6 images fom the first batch
    random_6 = random.sample(list(zip(imgs, preds, probs)), 6)

    plt.figure(figsize=(12, 15))

    for idx, (img, pred, prob) in enumerate(random_6, 1):
        img = np.array(img).reshape(224, 224, 3)
        plt.subplot(3, 2, idx)
        plt.imshow(img)
        plt.title(f"Prediction: {idx_to_class[pred]}, Prob: {prob:.2f}")
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
    print(f"The latest model path: {ckpt_path}")

    # sample prediction
    sample_prediction(model, data_module, ckpt_path)

# References

# 1. https://pytorch-lightning.readthedocs.io/en/latest/transfer_learning.html
# 1. https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# 1. https://pytorch.org/tutorials/beginner/saving_loading_models.html
# 1. https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=gEulmrbxwaYL
# 1. https://pytorch-lightning.readthedocs.io
# 1. https://github.com/PyTorchLightning/pytorch-lightning
# 1. https://www.youtube.com/watch?v=QHww1JH7IDU
# 1. https://pytorch-lightning.readthedocs.io/en/latest/
# 1. https://www.youtube.com/channel/UC8m-y0yAFJpX0hRvxH8wJVw/featured
