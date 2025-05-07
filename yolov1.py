import torch
import torch.nn as nn
import numpy as np
from yolov1_backbone import build_backbone
from yolov1_config import yolov1_cfg as cfg
from yolov1_neck import build_neck
from yolov1_head import build_head
from utility import multiclass_nms
class YOLOv1(nn.Module):
    def __init__(self,
                cfg,
                device,
                img_size=None,
                num_classes=20,
                conf_thresh=0.01,
                nms_thresh=0.5,
                trainable=False,
                deploy=False,
                nms_class_agnostic : bool = False):
        super(YOLOv1, self).__init__()
        self.cfg = cfg                      # 模型配置文件
        self.img_size = img_size            # 模型配置文件
        self.device = device                # cuda或者是cpu
        self.num_classes = num_classes      # 类别的数量
        self.trainable = trainable          # 训练集的标记
        self.conf_thresh = conf_thresh      # 得分阈值
        self.nms_thresh = nms_thresh        # NMS阈值
        self.stride = 32                    # 网络的最大步长
        self.deploy = deploy
        self.nms_class_agnostic = nms_class_agnostic

        # >>>>>>>>>> Backbone network >>>>>>>>>>>>>>
        #! To do: build our backbone network
        #? self.backbone
        self.backbone, feat_dim = build_backbone(cfg['backbone'], trainable&cfg['pretrained'])

        # >>>>>>>>>> Neck network >>>>>>>>>>>>>>
        #! To do: build our neck network
        #? self.neck
        self.neck = build_neck(cfg, feat_dim, out_dim=512)
        head_dim = self.neck.out_dim

        # >>>>>>>>>> Head network >>>>>>>>>>>>>>
        #! To do: build our head network
        #? self.head
        self.head = build_head(cfg, head_dim, head_dim, num_classes)

        # >>>>>>>>>> predict network >>>>>>>>>>>>>>
        #! To do: build our predict network
        #? self.pred
        self.obj_pred = nn.Conv2d(head_dim, 1, kernel_size=1)
        self.cls_pred = nn.Conv2d(head_dim, num_classes, kernel_size=1)
        self.reg_pred = nn.Conv2d(head_dim, 4, kernel_size=1)

    def create_grid(self, fmp_size):
        # To do:
        # 生成一个tensor: gridxy, 每个位置的元素是网格的坐标
        # 这个tensor将会在获得边界框参数的时候用到
        # 特征图宽度与高度
        ws, hs = fmp_size
        # grid_y, grid_x 都是二维矩阵
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        # grid_xy.shape = [grid_x.shape, grid_y.shape, 2]
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        # grid_xy.shape = [grid_x.shape * grid_y.shape, 2]
        grid_xy = grid_xy.view(-1, 2).to(self.device)
        return grid_xy

    def decode_boxes(self, pred, fmp_size):
        #? 将网络的输出tx, ty, tw, th四个量转换为bbox的(x1, y2), (x2, y2)
        """
            将txtytwth转换为常用的x1y1x2y2形式。
        """
        # 生成网格坐标矩阵
        grid_cell = self.create_grid(fmp_size)
        # 计算预测边界框的中心坐标和宽度高度
        pred_ctr = (torch.sigmoid(pred[..., :2]) + grid_cell) * self.stride
        pred_wh = torch.exp(pred[..., 2:]) * self.stride

        pred_x1y1 = pred_ctr - pred_wh * 0.5
        pred_x2y2 = pred_ctr + pred_wh * 0.5
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)
        return pred_box

    def postprocess(self, bboxes, scores):
        #? 后处理代码，包括阈值筛选和非极大值抑制
        """
            Input:
                bboxes: [HxW, 4]
                scores: [HxW, num_classes]
            Output:
                bboxes: [N, 4]
                score:  [N,]
                labels: [N,]
        """
        labels = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), labels)]

        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        #nms
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.nms_classes, self.nms_class_agnostic
        )
        return bboxes, scores, labels

    @torch.no_grad()
    def inference(self, x):
        #? 测试阶段不涉及反向传播
        feat = self.backbone(x)
        feat = self.neck(feat)
        cls_feat, reg_feat = self.head(feat)

        obj_pred = self.obj_pred(cls_feat)
        cls_pred = self.cls_pred(cls_feat)
        reg_pred = self.reg_pred(reg_feat)
        #fmp_size指的是特征图的尺寸
        fmp_size = obj_pred.shape[-2:]

        # 对 pred 的size做一些view调整，便于后续的处理
        # [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
        obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

        # 测试时，笔者默认batch是1，
        # 因此，我们不需要用batch这个维度，用[0]将其取走。
        obj_pred = obj_pred[0]       # [H*W, 1]
        cls_pred = cls_pred[0]       # [H*W, NC]
        reg_pred = reg_pred[0]       # [H*W, 4]
        scores = torch.sqrt(obj_pred.sigmoid() * cls_pred.sigmoid())
        # 解算边界框, 并归一化边界框: [H*W, 4]
        bboxes = self.decode_boxes(reg_pred, fmp_size)

        if self.deploy:
            outputs = torch.cat([bboxes, scores], dim=-1)
            return outputs
        else:
            scores = scores.cpu().numpy()
            bboxes = bboxes.cpu().numpy()
            bboxes, scores, labels = self.postprocess(bboxes, scores)
        return bboxes, scores, labels

    def forward(self, x, target=None):
        #? 训练时的前向推理代码
        if not self.trainable:
            return self.inference(x)
        else:
            feat = self.backbone(x)
            feat = self.neck(feat)
            cls_feat, reg_feat = self.head(feat)

            obj_pred = self.obj_pred(cls_feat)
            cls_pred = self.cls_pred(cls_feat)
            reg_pred = self.reg_pred(reg_feat)
            fmp_size = obj_pred.shape[-2:]

            obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

            box_pred = self.decode_boxes(reg_pred, fmp_size)

            outputs = {
                "pred_obj" : obj_pred,
                "pred_cls" : cls_pred,
                "pred_box" : box_pred,
                "stride" : self.stride,
                "fmp_size" : fmp_size
            }
            return outputs
