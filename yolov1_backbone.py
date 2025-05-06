import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101

class ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False):
        """zero_init_residual:若为True,则将残差块的最后一个BN层初始化为0,
   	    这样残差分支从0开始每一个残差分支,每一个残差块表现的像一个恒等映射
   	    根据论文:网络可提升0.2%~0.3%"""
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 3是输入通道 64是输出通道
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)       # inplace表示覆盖原来的值,应当可以节省内存
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def forward(self, x):
        """
        Input:
            x: (Tensor) -> [B, C, H, W]
        Output:
            c5: (Tensor) -> [B, C, H/32, W/32]
        """
        c1 = self.conv1(x)      #* [B, C, H/2, W/2]
        c1 = self.bn1(c1)       #* [B, C, H/2, W/2]
        c1 = self.relu(c1)      #* [B, C, H/2, W/2]
        c2 = self.maxpool(c1)   #* [B, C, H/4, W/4]

        c2 = self.layer1(c2)    #* [B, C, H/4, W/4]
        c3 = self.layer2(c2)    #* [B, C, H/8, W/8]
        c4 = self.layer3(c3)    #* [B, C, H/16, W/16]
        c5 = self.layer4(c4)    #* [B, C, H/32, W/32]
        return c5

def build_backbone(model_name='resnet18', pretrained=False):
    #? feat_dim 表示 特征向量的维度
    if model_name == 'resnet18':
        model = resnet18(pretrained)
        feat_dim = 512
    elif model_name == 'resnet34':
        model = resnet34(pretrained)
        feat_dim = 512
    elif model_name == 'resnet50':
        model = resnet50(pretrained)
        feat_dim = 2048
    elif model_name == 'resnet101':
        model = resnet101(pretrained)
        feat_dim = 2048
    return model, feat_dim
