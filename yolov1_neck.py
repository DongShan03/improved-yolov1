import torch.nn as nn
import torch
from utility import Conv

class SPPF(nn.Module):
    """
        This code referenced to https://github.com/ultralytics/yolov5
    """
    def __init__(self, in_dim, out_dim, expand_ratio=0.5, pooling_size=5):
        super().__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.out_dim = out_dim
        self.cv1 = Conv(in_dim, inter_dim, k=1)
        self.cv2 = Conv(inter_dim * 4, out_dim, k=1)
        self.m = nn.MaxPool2d(kernel_size=pooling_size, stride=1, padding=pooling_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)

        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

def build_neck(cfg, in_dim, out_dim):
    model = cfg['neck']
    print("====================")
    print('neck:{}'.format(model))
    # build neck
    if model == 'sppf':
        neck = SPPF(
            in_dim=in_dim,
            out_dim=out_dim,
            expand_ratio=cfg['expand_ratio'],
            pooling_size=cfg['pooling_size']
            # act_type norm_type 在这没用
        )
    return neck
