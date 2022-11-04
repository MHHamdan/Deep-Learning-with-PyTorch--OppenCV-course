import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionLoss(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + OHEMLoss(cls_preds, cls_targets).
        '''

        ################################################################
        # loc_loss
        ################################################################

        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.long().sum(1, keepdim=True)

        mask = pos.unsqueeze(2).expand_as(loc_preds)  # [N,#anchors,4]
        masked_loc_preds = loc_preds[mask].view(-1, 4)  # [#pos,4]
        masked_loc_targets = loc_targets[mask].view(-1, 4)  # [#pos,4]
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, reduction='none')
        loc_loss = loc_loss.sum() / num_pos.sum().float()

        ################################################################
        # cls_loss with OHEM
        ################################################################

        # Compute max conf across batch for hard negative mining
        batch_size, _ = cls_targets.size()
        batch_conf = cls_preds.view(-1, self.num_classes)
        cls_loss = F.cross_entropy(batch_conf, cls_targets.view(-1), ignore_index=-1, reduction='none')
        cls_loss = cls_loss.view(batch_size, -1)

        # Hard Negative Mining
        # filter out pos boxes (pos = cls_targets > 0) for now.
        pos_cls_loss = cls_loss[pos]
        
        # In OHEM, we have to select only those background labels (0) that have been failed with 
        # a very high margin (lets we will choose three times (negpos_ratio = 3) of the object labels (>=1)). 
        
        # To paly around background labels, let's make zero loss to object labels.
        cls_loss[pos] = 0 
        
        # Let's find indices of decreasing order of loss (which ground truth is background). 
        _, loss_idx = cls_loss.sort(1, descending=True)
        
        # If we sort (in increasing order) the above indices, indices correspond to the sorting will 
        # give a ranking (along dimension 1) of the original loss matrix. 
        _, idx_rank = loss_idx.sort(1)
        
        # Let's understand by example. As all operations are along axis 1, taking 1-d example will be sufficient.

        # cls_loss = [5, 2, 9, 6,  8]

        # _, loss_idx = cls_loss.sort(descending=True)
        # loss_idx = [2, 4, 3, 0, 1]

        # _, idx_rank = loss_idx.sort()
        # idx_rank = [3, 4, 0, 2, 1]
        
        # Have a look, idx_rank has the ranking of cls_loss.

        negpos_ratio = 3
        
        # We have decided we will take the negative class count three times of the positive class. 

        # If we do it blindly, in the case of not a positive class in the image, we will end up missing 
        # all the negative class also. So let's clamp minimum to 1. 
        # Although maximum clamping is not required here,  let fix to maximum index. 

        num_neg = torch.clamp(negpos_ratio * num_pos, min=1, max=pos.size(1) - 1)

        neg = idx_rank < num_neg.expand_as(idx_rank)
        neg_cls_loss = cls_loss[neg]

        cls_loss = (pos_cls_loss.sum() + neg_cls_loss.sum()) / num_pos.sum().float()
        
        # The magnitude of cross-entropy loss is much more than L2, L1, and smooth L1. 
        # So it is better to take a weighted loss. Here we have chosen twenty times of 
        # lower magnitude loss and one time of higher magnitude loss.

        return 20 * loc_loss, cls_loss
