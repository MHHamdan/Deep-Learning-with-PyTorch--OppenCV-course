import random
import numpy as np
import torch

from .configuration import SystemConfig, TrainerConfig, DataloaderConfig


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
        batch_size_to_set = 1
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


def random_flip(img, bbox):
    '''Randomly flip the given image tensor

    Args:
        img (tensor): image tensor. shape (3, height, width)
        bbox (tensor): bounding box tensor

    Returns:
        img (tensor): randomaly fliped image tensor.
        bbox (tensor): randomaly fliped bounding box tensor
    '''
    if random.random() < 0.5:

        _, width = img.shape[-2:]
        img = img.flip(-1)
        bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
        
    return img, bbox


def collate_fn(batch):
    return tuple(zip(*batch))
