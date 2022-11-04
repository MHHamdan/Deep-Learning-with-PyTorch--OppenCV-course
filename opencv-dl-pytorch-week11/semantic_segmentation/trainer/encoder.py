import math

import numpy as np
import torch


class DataEncoder:
    def __init__(self, input_size):
        self.input_size = input_size
        self.anchor_areas = [8 * 8, 16 * 16., 32 * 32., 64 * 64., 128 * 128]  # p3 -> p7
        self.aspect_ratios = [0.5, 1, 2]
        self.scales = [1, pow(2, 1 / 3.), pow(2, 2 / 3.)]
        num_fms = len(self.anchor_areas)
        fm_sizes = [math.ceil(self.input_size[0] / pow(2., i + 3)) for i in range(num_fms)]
        self.anchor_boxes = []
        for i, fm_size in enumerate(fm_sizes):
            anchors = generate_anchors(self.anchor_areas[i], self.aspect_ratios, self.scales)
            anchor_grid = generate_anchor_grid(input_size, fm_size, anchors)
            self.anchor_boxes.append(anchor_grid)
        self.anchor_boxes = torch.cat(self.anchor_boxes, 0)
        self.classes = ["__background__", "person"]

    def encode(self, boxes, classes):
        iou = compute_iou(boxes, self.anchor_boxes)
        iou, ids = iou.max(1)
        loc_targets = encode_boxes(boxes[ids], self.anchor_boxes)
        cls_targets = classes[ids]
        cls_targets[iou < 0.5] = -1
        cls_targets[iou < 0.4] = 0

        return loc_targets, cls_targets

    def decode(self, loc_pred, cls_pred, cls_threshold=0.7, nms_threshold=0.3):
        all_boxes = [[] for _ in range(len(loc_pred))]  # batch_size

        for sample_id, (boxes, scores) in enumerate(zip(loc_pred, cls_pred)):
            boxes = decode_boxes(boxes, self.anchor_boxes)

            conf = scores.softmax(dim=1)
            sample_boxes = [[] for _ in range(len(self.classes))]
            for class_idx, class_name in enumerate(self.classes):
                if class_name == '__background__':
                    continue
                class_conf = conf[:, class_idx]
                ids = (class_conf > cls_threshold).nonzero().squeeze()
                ids = [ids.tolist()]
                keep = compute_nms(boxes[ids], class_conf[ids], threshold=nms_threshold)

                conf_out, top_ids = torch.sort(class_conf[ids][keep], dim=0, descending=True)
                boxes_out = boxes[ids][keep][top_ids]

                boxes_out = boxes_out.cpu().numpy()
                conf_out = conf_out.cpu().numpy()

                c_dets = np.hstack((boxes_out, conf_out[:, np.newaxis])).astype(np.float32, copy=False)
                c_dets = c_dets[c_dets[:, 4].argsort()]
                sample_boxes[class_idx] = c_dets

            all_boxes[sample_id] = sample_boxes

        return all_boxes

    def get_num_anchors(self):
        return len(self.aspect_ratios) * len(self.scales)


def decode_boxes(deltas, anchors):
    if torch.cuda.is_available():
        anchors = anchors.cuda()
    anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
    anchors_ctr = anchors[:, :2] + 0.5 * anchors_wh
    pred_ctr = deltas[:, :2] * anchors_wh + anchors_ctr
    pred_wh = torch.exp(deltas[:, 2:]) * anchors_wh
    return torch.cat([pred_ctr - 0.5 * pred_wh, pred_ctr + 0.5 * pred_wh - 1], 1)


def encode_boxes(boxes, anchors):
    anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
    anchors_ctr = anchors[:, :2] + 0.5 * anchors_wh
    boxes_wh = boxes[:, 2:] - boxes[:, :2] + 1
    boxes_ctr = boxes[:, :2] + 0.5 * boxes_wh
    return torch.cat([(boxes_ctr - anchors_ctr) / anchors_wh, torch.log(boxes_wh / anchors_wh)], 1)


def generate_anchor_grid(input_size, fm_size, anchors):
    grid_size = input_size[0] / fm_size
    x, y = torch.meshgrid(torch.arange(0, fm_size) * grid_size, torch.arange(0, fm_size) * grid_size)
    anchors = anchors.view(-1, 1, 1, 4)
    xyxy = torch.stack([x, y, x, y], 2).float()
    boxes = (xyxy + anchors).permute(2, 1, 0, 3).contiguous().view(-1, 4)
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, input_size[0])
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, input_size[1])
    return boxes


def generate_anchors(anchor_area, aspect_ratios, scales):
    anchors = []
    for scale in scales:
        for ratio in aspect_ratios:
            h = round(math.sqrt(anchor_area) / ratio)
            w = round(ratio * h)
            x1 = (math.sqrt(anchor_area) - scale * w) * 0.5
            y1 = (math.sqrt(anchor_area) - scale * h) * 0.5
            x2 = (math.sqrt(anchor_area) + scale * w) * 0.5
            y2 = (math.sqrt(anchor_area) + scale * h) * 0.5
            anchors.append([x1, y1, x2, y2])
    return torch.Tensor(anchors)


def compute_iou(src, dst):
    p1 = torch.max(dst[:, None, :2], src[:, :2])
    p2 = torch.min(dst[:, None, 2:], src[:, 2:])
    inter = torch.prod((p2 - p1 + 1).clamp(0), 2)
    src_area = torch.prod(src[:, 2:] - src[:, :2] + 1, 1)
    dst_area = torch.prod(dst[:, 2:] - dst[:, :2] + 1, 1)
    iou = inter / (dst_area[:, None] + src_area - inter)
    return iou


def compute_nms(boxes, conf, threshold=0.5):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    _, order = conf.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0] if order.numel() > 1 else order.item()
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i].item())
        yy1 = y1[order[1:]].clamp(min=y1[i].item())
        xx2 = x2[order[1:]].clamp(max=x2[i].item())
        yy2 = y2[order[1:]].clamp(max=y2[i].item())

        w = (xx2 - xx1 + 1).clamp(min=0)
        h = (yy2 - yy1 + 1).clamp(min=0)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        ids = (ovr <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]
    return torch.LongTensor(keep)
