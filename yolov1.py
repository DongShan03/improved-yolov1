import torch
import torch.nn as nn
from yolov1_backbone import build_backbone
from yolov1_config import yolov1_cfg as cfg
from yolov1_neck import build_neck
from yolov1_head import build_head
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
        pass

    def decode_boxes(self, pred, fmp_size):
        #? 将网络的输出tx, ty, tw, th四个量转换为bbox的(x1, y2), (x2, y2)
        pass

    def postprocess(self, bboxes, scores):
        #? 后处理代码，包括阈值筛选和非极大值抑制
        pass

    @torch.no_grad()
    def inference(self, x):
        #? 测试阶段不涉及反向传播
        pass

    def forward(self, x, target=None):
        #? 训练时的前向推理代码
        pass
