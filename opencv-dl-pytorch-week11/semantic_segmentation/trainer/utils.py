import os
import sys
import random
import shutil
import tempfile

import cv2
import git
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from .configuration import SystemConfig, TrainerConfig, DataloaderConfig


def download_git_folder(git_url, src_folder, dst_folder):
    """ Download folder from remote git repo

    Arguments:
        git_url (string): url of remote git repository.
        src_folder (string): path to required folder (related of git repo root).
        dst_folder (string): destination path for required folder (local path).
    """
    class Progress(git.RemoteProgress):
        def update(self, op_code, cur_count, max_count=None, message=''):  # pylint: disable=unused-argument
            sys.stdout.write('\r')
            sys.stdout.write('Download: {}'.format(message).ljust(80, ' '))
            sys.stdout.flush()

    tmp = tempfile.mkdtemp()
    git.Repo.clone_from(git_url, tmp, branch='master', depth=1, progress=Progress())
    shutil.move(os.path.join(tmp, src_folder), dst_folder)
    shutil.rmtree(tmp)


def init_semantic_segmentation_dataset(data_path, imgs_folder="train", masks_folder="trainannot"):
    """ Prepare Semantic Segmentation dataset

    Arguments:
        data_path (string): path to dataset.
        imgs_folder (string): path to images (related of the data_path).
        masks_folder (string): path to masks (related of the data_path).

    Initialise dataset as a list of pairs {"image": <path_to_image>, "mask": <path_to_mask>}.
    """
    names = os.listdir(os.path.join(data_path, imgs_folder))
    dataset = []
    for name in names:
        dataset.append({
            "image": os.path.join(data_path, imgs_folder, name),
            "mask": os.path.join(data_path, masks_folder, name)
        })
    return dataset


def draw_semantic_segmentation_samples(dataset, n_samples=3):
    """ Draw samples from semantic segmentation dataset.

    Arguments:
        dataset (iterator): dataset class.
        plt (matplotlib.pyplot): canvas to show samples.
        n_samples (int): number of samples to visualize.
    """
    fig, ax = plt.subplots(nrows=n_samples, ncols=2, sharey=True, figsize=(10, 10))
    for i, sample in enumerate(dataset):
        if i >= n_samples:
            break
        ax[i][0].imshow(sample["image"])
        ax[i][0].set_xlabel("image")
        ax[i][0].set_xticks([])
        ax[i][0].set_yticks([])

        ax[i][1].imshow(sample["mask"])
        ax[i][1].set_xlabel("mask")
        ax[i][1].set_xticks([])
        ax[i][1].set_yticks([])

    plt.tight_layout()
    plt.gcf().canvas.draw()
    plt.show()
    plt.close(fig)


def draw_semantic_segmentation_batch(images, masks_gt, masks_pred=None, n_samples=3):
    """ Draw batch from semantic segmentation dataset.

    Arguments:
        images (torch.Tensor): batch of images.
        masks_gt (torch.LongTensor): batch of ground-truth masks.
        plt (matplotlib.pyplot): canvas to show samples.
        masks_pred (torch.LongTensor, optional): batch of predicted masks.
        n_samples (int): number of samples to visualize.
    """
    nrows = min(images.size(0), n_samples)
    ncols = 2 if masks_pred is None else 3
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, figsize=(10, 10))
    for i in range(nrows):
        img = images[i].permute(1, 2, 0).detach().cpu().numpy()
        img = np.clip(img, 0, 1)
        ax[i][0].imshow(img)
        ax[i][0].set_xlabel("image")
        ax[i][0].set_xticks([])
        ax[i][0].set_yticks([])
        gt_mask = masks_gt[i].detach().cpu().numpy()
        ax[i][1].imshow(gt_mask)
        ax[i][1].set_xlabel("ground-truth mask")
        ax[i][1].set_xticks([])
        ax[i][1].set_yticks([])
        if masks_pred is not None:
            pred = masks_pred[i].detach().cpu().numpy()
            ax[i][2].imshow(pred)
            ax[i][2].set_xlabel("predicted mask")
            ax[i][2].set_xticks([])
            ax[i][2].set_yticks([])

    plt.tight_layout()
    plt.gcf().canvas.draw()
    plt.show()
    plt.close(fig)


def get_camvid_dataset_parameters(data_path, dataset_type="train", transforms=None):
    """ Get CamVid parameters compatible with DanseData dataset class.

    Arguments:
        data_path (string): path to dataset folder.
        dataset_type (string): dataset type (train, test or val).
        transforms (callable, optional): A function/transform that takes in a sample
            and returns a transformed version.

    Returns:
        dictionary with parameters of CamVid dataset.
    """
    return {
        "data_path":
            os.path.join(data_path, "CamVid"),
        "images_folder":
            dataset_type,
        "masks_folder":
            dataset_type + "annot",
        "num_classes":
            11,
        "transforms":
            transforms,
        "dataset_url":
            "https://github.com/alexgkendall/SegNet-Tutorial.git",
        "dataset_folder":
            "CamVid",
        "class_names": [
            'sky', 'building', 'pole', 'road', 'pavement', 'tree', 'signsymbol', 'fence', 'car', 'pedestrian',
            'bicyclist', 'unlabelled'
        ]
    }


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, count=1):
        self.val = val
        self.sum += val * count
        self.count += count
        self.avg = self.sum / self.count


class DeviceManager:
    """ Helper class to auto upload variables to CUDA.

    Arguments:
        use_cuda (bool): if True then variables will be upload to GPU.
    """
    def __init__(self, use_cuda=True):
        self.use_cuda = use_cuda

    def __call__(self, var):
        """ Upload variable to default device.

        Arguments:
            var (torch.Module or torch.Tensor): input variable.

        Returns:
            Returns input variable uploaded to select default device.
        """
        return var.cuda() if self.use_cuda else var


def patch_configs(epoch_num_to_set=TrainerConfig.epoch_num, batch_size_to_set=DataloaderConfig.batch_size):
    """ Patches configs if cuda is not available

    Returns:
        returns patched dataloader_config and trainer_config

    """
    # default experiment params
    num_workers_to_set = DataloaderConfig.num_workers

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        batch_size_to_set = 16
        num_workers_to_set = 2
        epoch_num_to_set = 1

    dataloader_config = DataloaderConfig(batch_size=batch_size_to_set, num_workers=num_workers_to_set)
    trainer_config = TrainerConfig(device=device, epoch_num=epoch_num_to_set, progress_bar=True)
    return dataloader_config, trainer_config


def setup_system(system_config: SystemConfig) -> None:
    torch.manual_seed(system_config.seed)
    np.random.seed(system_config.seed)
    random.seed(system_config.seed)
    torch.set_printoptions(precision=10)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(system_config.seed)
        torch.backends.cudnn_benchmark_enabled = system_config.cudnn_benchmark_enabled
        torch.backends.cudnn.deterministic = system_config.cudnn_deterministic


def fit_class_number(model: nn.Module, num_classes: int) -> nn.Module:
    filters_num = model.fc.in_features
    model.fc = nn.Linear(filters_num, num_classes)
    return model


def draw_samples(data, rows, cols):
    assert len(data) >= rows * cols
    fig, ax = plt.subplots(nrows=rows, ncols=cols, sharey=True, figsize=(10, 10))
    iterator = iter(data)
    for idx in range(rows * cols):
        img, _ = next(iterator)
        i = idx // cols
        j = idx % cols
        ax[i][j].axis("off")
        ax[i][j].imshow(img, interpolation="nearest")
    plt.tight_layout()
    fig.canvas.draw()
    plt.show()


def resize(img, boxes, size, max_size=1000):
    '''Resize the input cv2 image to the given size.

    Args:
      img: (cv2) image to be resized.
      boxes: (tensor) object boxes, sized [#ojb,4].
      size: (tuple or int)
        - if is tuple, resize image to the size.
        - if is int, resize the shorter side to the size while maintaining the aspect ratio.
      max_size: (int) when size is int, limit the image longer size to max_size.
                This is essential to limit the usage of GPU memory.
    Returns:
      img: (cv2) resized image.
      boxes: (tensor) resized boxes.
    '''
    height, width, _ = img.shape
    if isinstance(size, int):
        size_min = min(width, height)
        size_max = max(width, height)
        scale_w = scale_h = float(size) / size_min
        if scale_w * size_max > max_size:
            scale_w = scale_h = float(max_size) / size_max
        new_width = int(width * scale_w + 0.5)
        new_height = int(height * scale_h + 0.5)
    else:
        new_width, new_height = size
        scale_w = float(new_width) / width
        scale_h = float(new_height) / height

    return cv2.resize(img, (new_height, new_width)), \
           boxes * torch.Tensor([scale_w, scale_h, scale_w, scale_h])


def random_flip(img, boxes):
    '''Randomly flip the given cv2 Image.

    Args:
        img: (cv2) image to be flipped.
        boxes: (tensor) object boxes, sized [#ojb,4].

    Returns:
        img: (cv2) randomly flipped image.
        boxes: (tensor) randomly flipped boxes.
    '''
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        width = img.shape[1]
        xmin = width - boxes[:, 2]
        xmax = width - boxes[:, 0]
        boxes[:, 0] = xmin
        boxes[:, 2] = xmax
    return img, boxes
