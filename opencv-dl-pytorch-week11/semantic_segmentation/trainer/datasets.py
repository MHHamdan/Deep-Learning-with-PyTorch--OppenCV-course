import os
import sys
import glob

import cv2
import numpy as np
import torch

from torch.utils.data import Dataset

from .utils import resize, random_flip
from .encoder import DataEncoder


class TinyImageNetDataset(Dataset):
    '''
    root_dir: data set root directory
    mode: train|test|val
    transform: transformations series
    '''

    extension: str = "JPEG"  # dataset images extension definition
    num_images_per_class: int = 500  # number of dataset images per class
    classes_list: str = "wnids.txt"  # file, which contains dataset classes list
    val_annotation_classes: str = "val/val_annotations.txt"  # validation images and classes mapping file
    num_class: int = 200

    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = os.path.expanduser(root_dir)
        self.mode = mode
        self.transform = transform
        self.sub_dir = os.path.join(root_dir, self.mode)
        self.image_paths = sorted(
            glob.iglob(os.path.join(self.sub_dir, '**', '*.{}'.format(self.extension)), recursive=True)
        )
        self.labels = {}
        self.images = []

        with open(os.path.join(self.root_dir, self.classes_list), 'r') as file_to_read:
            # sorted list of classes: ['n01443537', 'n01629819', 'n01641577' ...]
            self.label_texts = sorted([text.strip() for text in file_to_read.readlines()])
        # classes names and their ordinal number mapping {'n01443537': 0, ...}
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

        # labels filling
        if self.mode == 'train':
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(self.num_images_per_class):
                    # map images from one class with its class ordinal number: {..., 'n01443537_178.JPEG': 0, ...}
                    self.labels['{0}_{1}.{2}'.format(label_text, cnt, self.extension)] = i
        elif self.mode == 'val':
            with open(os.path.join(self.root_dir, self.val_annotation_classes), 'r') as file_to_read:
                for line in file_to_read.readlines():
                    terms = line.split('\t')
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        file_path = self.image_paths[i]
        img = self.process_image(file_path)

        if self.mode != 'test':
            return {'image': img, 'target': self.labels[os.path.basename(file_path)]}
        return {'image': img, 'target': None}

    def process_image(self, path):
        img = cv2.imread(path)
        if img is None or np.prod(img.shape) == 0:
            print('cannot load image from path: ', path)
            sys.exit(-1)
        img = img[..., ::-1]
        if self.transform:
            img = self.transform(image=img)['image']
        return img


class ListDataset(Dataset):
    def __init__(self, root_dir, list_file, classes, mode, transform, input_size):
        '''
        Args:
          root_dir: (str) ditectory to images.
          list_file: (str) path to index file.
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
        '''
        self.root_dir = root_dir
        self.classes = classes
        self.mode = mode
        self.transform = transform
        self.input_size = input_size

        self.fnames = []
        self.boxes = []
        self.labels = []

        list_file_path = os.path.join(root_dir, list_file)
        with open(list_file_path) as f:
            lines = f.readlines()
            self.num_samples = len(lines)

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            num_boxes = (len(splited) - 1) // 5
            box = []
            label = []
            for i in range(num_boxes):
                xmin = splited[1 + 5 * i]
                ymin = splited[2 + 5 * i]
                xmax = splited[3 + 5 * i]
                ymax = splited[4 + 5 * i]
                class_label = splited[5 + 5 * i]
                box.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                label.append(int(class_label))
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        # Load image and boxes.
        path = os.path.join(self.root_dir, self.fnames[idx])
        img = cv2.imread(path)
        if img is None or np.prod(img.shape) == 0:
            print('cannot load image from path: ', path)
            sys.exit(-1)

        img = img[..., ::-1]

        boxes = self.boxes[idx].clone()
        labels = self.labels[idx]
        size = self.input_size

        # Resize & Flip
        img, boxes = resize(img, boxes, (size, size))
        if self.mode == 'train':
            img, boxes = random_flip(img, boxes)
        # Data augmentation.
        img = np.array(img)
        if self.transform:
            img = self.transform(image=img)['image']

        return img, boxes, labels

    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]

        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, w, h)
        encoder = DataEncoder((w, h))
        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            loc_target, cls_target = encoder.encode(boxes[i], labels[i])
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets)

    def __len__(self):
        return self.num_samples
