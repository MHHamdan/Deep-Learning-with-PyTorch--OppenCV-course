import os
import sys

from PIL import Image
import torch
from torchvision.transforms import functional as F

from torch.utils.data import Dataset

from .utils import random_flip


class ListDataset(Dataset):
    def __init__(self, csv_path, train=False, transform=None):
        '''
        Args:
          csv_path: (str) ditectory to images.
          train: (boolean) True if train else False.
          transform: ([transforms]) image transforms.
        '''
        
        self.transform = transform
        self.train = train

        self.fnames = []
        self.boxes = []
        self.labels = []

        with open(csv_path) as f:
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
          targets: (dict) location and bbox targets.
        '''
        # Load image and boxes.
        img = Image.open(self.fnames[idx]).convert("RGB")
        # transform PIL to tensor
        img = F.to_tensor(img)
        
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx]        
            
        # Data augmentation.
        if self.transform is not None:
            img = self.transform(img)
            
        if self.train:
            img, boxes = random_flip(img, boxes)
            
        target = dict()
        target["boxes"] = boxes
        target["labels"] = labels

        return img, target

    def __len__(self):
        return self.num_samples
