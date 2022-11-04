import torch
import torch.nn as nn

from fpn import FPN


class Detector(nn.Module):
    num_anchors = 9

    def __init__(self, num_classes=2):
        super(Detector, self).__init__()
        self.fpn = FPN()
        self.num_classes = num_classes
        self.loc_head = self._make_head(self.num_anchors * 4)
        self.cls_head = self._make_head(self.num_anchors * self.num_classes)

    def forward(self, x):
        fms = self.fpn(x)
        loc_preds = []
        cls_preds = []
        for feature_map in fms:
            loc_pred = self.loc_head(feature_map)
            cls_pred = self.cls_head(feature_map)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view(
                x.size(0), -1, 4
            )  # [N, 9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(
                x.size(0), -1, self.num_classes
            )  # [N,9*20,H,W] -> [N,H,W,9*20] -> [N,H*W*9,20]
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
            
        
        return torch.cat(loc_preds, 1), torch.cat(cls_preds, 1)

    @staticmethod
    def _make_head(out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(64, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)
