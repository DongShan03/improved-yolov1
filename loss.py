from matcher import YoloMatcher
import torch.nn.functional as F
import torch
from box_ops import get_ious
from distributed_utils import get_world_size, is_dist_avail_and_initialized
class Criterion(object):
    def __init__(self, cfg, device, num_classes=80):
        self.cfg = cfg
        self.device = device
        self.num_classes = num_classes
        self.loss_obj_weight = cfg["loss_obj_weight"]
        self.loss_cls_weight  = cfg["loss_cls_weight"]
        self.loss_box_weight  = cfg["loss_box_weight"]

        self.matcher = YoloMatcher(num_classes=num_classes)

    def loss_objectness(self, pred_obj, gt_obj):
        loss_obj = \
            F.binary_cross_entropy_with_logits(pred_obj,
                                                gt_obj,
                                                reduction='none')
        return loss_obj

    def loss_classes(self, pred_cls, gt_label):
        loss_cls = \
            F.binary_cross_entropy_with_logits(pred_cls,
                                                gt_label,
                                                reduce='none')
        return loss_cls

    def loss_bboxes(self, pred_box, gt_box):
        ious = get_ious(pred_box, gt_box, box_mode="xyxy", iou_type="giou")
        loss_box = 1.0 - ious
        return loss_box

    def __call__(self, outputs, targets, epoch=0):
        device = outputs['pred_cls'][0].device
        stride = outputs['stride']
        fmp_size = outputs['fmp_size']
        (
            gt_objectness,
            gt_classes,
            gt_bboxes,
        ) = self.matcher(fmp_size=fmp_size,
                         stride=stride,
                         targets=targets)

        # List[B, M, C] -> [B, M, C] -> [BM, C]
        pred_obj = outputs['pred_obj'].view(-1)
        pred_cls = outputs['pred_cls'].view(-1, self.num_classes)
        pred_box = outputs['pred_box'].view(-1, 4)

        gt_objectness = gt_objectness.view(-1).to(device).float()
        gt_classes = gt_classes.view(-1, self.num_classes).to(device).float()
        gt_bboxes = gt_bboxes.view(-1, 4).to(device).float()

        pos_masks = (gt_objectness > 0)
        num_fgs = pos_masks.sum()

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_fgs)
        num_fgs = (num_fgs / get_world_size()).clamp(1.0)

        # cls loss
        pred_cls_pos = pred_cls[pos_masks]
        gt_classes_pos = gt_classes[pos_masks]
        loss_cls = self.loss_classes(pred_cls_pos, gt_classes_pos)
        loss_cls = loss_cls.sum() / num_fgs

        # obj loss
        loss_obj = self.loss_objectness(pred_obj, gt_objectness)
        loss_obj = loss_obj.sum() / num_fgs

        # box loss
        pred_box_pos = pred_box[pos_masks]
        gt_bboxes_pos = gt_bboxes[pos_masks]
        loss_box = self.loss_bboxes(pred_box_pos, gt_bboxes_pos)
        loss_box = loss_box.sum() / num_fgs

        # total loss
        losses = self.loss_obj_weight * loss_obj + \
                    self.loss_cls_weight * loss_cls + \
                    self.loss_box_weight * loss_box

        loss_dict = dict(
                loss_obj = loss_obj,
                loss_cls = loss_cls,
                loss_box = loss_box,
                losses = losses
        )

        return loss_dict
