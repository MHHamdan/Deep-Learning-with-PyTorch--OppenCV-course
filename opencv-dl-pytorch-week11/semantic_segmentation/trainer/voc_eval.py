from collections import defaultdict

import numpy as np
import torch


def calculate_overlap(BBGT, bb):
    inter_xmin = np.maximum(BBGT[:, 0], bb[0])
    inter_ymin = np.maximum(BBGT[:, 1], bb[1])
    inter_xmax = np.minimum(BBGT[:, 2], bb[2])
    inter_ymax = np.minimum(BBGT[:, 3], bb[3])
    inter_width = np.maximum(inter_xmax - inter_xmin + 1., 0.)
    inter_height = np.maximum(inter_ymax - inter_ymin + 1., 0.)
    inters = inter_width * inter_height

    union = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + (BBGT[:, 2] - BBGT[:, 0] + 1.) *
             (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

    overlaps = inters / union
    ovmax = np.max(overlaps)
    jmax = np.argmax(overlaps)
    return ovmax, jmax


class VOCEvaluator():
    @staticmethod
    def voc_ap(recall, precision, use_07_metric=False):
        """ ap = voc_ap(recall, precision, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for treshold in np.arange(0., 1.1, 0.1):
                if np.sum(recall >= treshold) == 0:
                    p = 0
                else:
                    p = np.max(precision[recall >= treshold])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mean_recall = np.concatenate(([0.], recall, [1.]))
            mean_precision = np.concatenate(([0.], precision, [0.]))

            # compute the precision envelope
            for i in range(mean_precision.size - 1, 0, -1):
                mean_precision[i - 1] = np.maximum(mean_precision[i - 1], mean_precision[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mean_recall[1:] != mean_recall[:-1])[0]

            # and sum (\Delta recall) * precision
            ap = np.sum((mean_recall[i + 1] - mean_recall[i]) * mean_precision[i + 1])
        return ap

    def __init__(self, classes, overlap_thresh=0.5, use_07_metric=False):
        self.classes = classes
        self.gt_counter_per_class = defaultdict(int)
        self.overlap_thresh = overlap_thresh
        self.class_recs = {}
        self.imagenames = defaultdict(list)
        self.boxes = defaultdict(list)
        self.scores = defaultdict(list)
        self.use_07_metric = use_07_metric
        self.s_counter = 0
        for class_idx in range(len(self.classes)):
            if self.classes[class_idx] == '__background__':
                continue
            self.gt_counter_per_class[class_idx] = 0
            self.imagenames[class_idx] = []
            self.boxes[class_idx] = []
            self.scores[class_idx] = []
            self.class_recs[class_idx] = {}

    def add_sample(self, all_dets, gt_boxes):
        self.s_counter += 1
        image_name = self.s_counter

        for class_idx in range(len(self.classes)):
            if self.classes[class_idx] == '__background__':
                continue
            c_dets = all_dets[class_idx]

            if c_dets.size == 0:
                continue
            pos_num = np.count_nonzero(gt_boxes[:, 4] == float(class_idx))
            self.gt_counter_per_class[class_idx] += pos_num
            record = gt_boxes[np.where(gt_boxes[:, -1] == class_idx)[0]]
            bbox = record[:, :4]
            det = np.full((len(record), ), False)
            self.class_recs[class_idx][image_name] = {'bbox': bbox, 'det': det}
            image_names = np.full((len(c_dets), ), image_name)
            self.imagenames[class_idx].append(image_names)
            self.boxes[class_idx].append(c_dets[:, :4])
            self.scores[class_idx].append(c_dets[:, -1])

    def evaluate(self):
        aps = []
        for class_idx in range(len(self.classes)):
            if self.classes[class_idx] == '__background__':
                continue

            if len(self.boxes[class_idx]) == 0:
                continue

            BB = np.concatenate(self.boxes[class_idx])
            confidence = np.concatenate(self.scores[class_idx])
            sorted_ind = np.argsort(-confidence, kind='stable')
            BB = BB[sorted_ind, :]
            image_ids = np.concatenate(self.imagenames[class_idx])[sorted_ind]
            num_images = len(image_ids)
            tp = np.zeros(num_images)
            fp = np.zeros(num_images)
            for sample in range(num_images):
                record = self.class_recs[class_idx][image_ids[sample]]
                bb = BB[sample, :]

                ovmax = -np.inf
                BBGT = record['bbox']

                if torch.cuda.is_available():
                    BBGT = BBGT.cpu().numpy()

                if BBGT.size > 0:
                    ovmax, jmax = calculate_overlap(BBGT, bb)

                if ovmax > self.overlap_thresh:
                    if not record['det'][jmax]:
                        tp[sample] = 1.
                        record['det'][jmax] = 1
                    else:
                        fp[sample] = 1.
                else:
                    fp[sample] = 1.

            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            recall = tp / float(self.gt_counter_per_class[class_idx])
            precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = self.voc_ap(recall, precision, use_07_metric=self.use_07_metric) * 100
            aps.append(ap)
        return aps
