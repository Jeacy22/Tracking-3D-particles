import math
import torch.nn as nn
from .networks.ResNet import resnet101,resnet_Head,resnet_Decoder,resnet50
from .networks.ResNetX import resnet_DecoderX,resnet50_x
from .networks.ConvNext import convnext_small,convnext_Decoder,convnext_Head
from .networks.DLA34 import get_pose_net
from timm.models.layers import trunc_normal_

class CenterNet_Resnet(nn.Module):
    def __init__(self,num_classes= 2, pretrained = False):
        super(CenterNet_Resnet,self).__init__()
        self.pretrained = pretrained
        self.backbone = resnet101(pretrained=pretrained)
        self.decoder = resnet_Decoder(2048)
        self.head = resnet_Head(channel=64,num_classes=num_classes)

        self._init_weight()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad=False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad=True


    def _init_weight(self):
        if not self.pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        self.head.cls_head[-1].weight.data.fill_(0)
        self.head.cls_head[-1].bias.data.fill_(-2.19)


    def forward(self, x):
        feat = self.backbone(x)
        return self.head(self.decoder(feat))


class CenterNet_ConvNext(nn.Module):
    def __init__(self,num_classes= 2, pretrained = False):
        super(CenterNet_ConvNext,self).__init__()
        self.pretrained = pretrained
        self.backbone = convnext_small(pretrained=pretrained)
        self.decoder = convnext_Decoder(768)
        self.head = convnext_Head(channel=64,num_classes=num_classes)
        self._init_weight()


    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad=False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad=True

    def _init_weight(self):
        if not self.pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=0.1)
                    nn.init.constant_(m.bias, 0)

        self.head.cls_head[-1].weight.data.fill_(0)
        self.head.cls_head[-1].bias.data.fill_(-2.19)


    def forward(self, x):
        feat = self.backbone(x)

        return self.head(self.decoder(feat))



class CenterNet_Dla34(nn.Module):
    def __init__(self, num_classes):
        super(CenterNet_Dla34, self).__init__()
        self.num_classes = num_classes
        self.heads = {'hm': self.num_classes,
                      'depth': 1,
                      'reg': 2}

        self.model = get_pose_net(num_layers=34, heads=self.heads)


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        feat = self.model(x)
        return feat[0]['hm'].sigmoid_(),feat[0]['depth'],feat[0]['reg']





class CenterNet_ResnetX(nn.Module):
    def __init__(self, num_classes,pretrained):
        super(CenterNet_ResnetX, self).__init__()
        self.pretrained = pretrained
        self.backbone = resnet50_x(pretrained=pretrained)
        self.decoder = resnet_DecoderX(2048)
        self.head = convnext_Head(channel=64, num_classes=num_classes)

        self._init_weight()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad=False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad=True


    def _init_weight(self):
        if not self.pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        self.head.cls_head[-1].weight.data.fill_(0)
        self.head.cls_head[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(self.decoder(feat))







































