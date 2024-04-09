import torch.nn as nn
from model.RESNET import resnet_Head,resnet_Decoder,ResNet50,ResNet101
from model.RESNETX import ResNetn50,resnet_Head,resnet_DecoderX
from model.DLA34 import get_pose_net
import math


def init_weight_backbone(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def init_weight_head(objectn):
    objectn.cls_head[-1].weight.data.fill_(0)
    objectn.cls_head[-1].bias.data.fill_(-2.19)
    objectn.depth_head[-1].weight.data.fill_(0)
    objectn.depth_head[-1].bias.data.fill_(-2.19)
    objectn.reg_head[-1].weight.data.fill_(0)
    objectn.reg_head[-1].bias.data.fill_(-2.19)



class CenterNet_ResNet(nn.Module):
    def __init__(self,numclass,pretrained,resnet_flag):
        super(CenterNet_ResNet,self).__init__()
        self.pretrained = pretrained
        self.num_class = numclass
        self.decoder = resnet_Decoder(2048)
        self.head = resnet_Head(channel=64,num_classes=numclass)
        self.resnet_flag = resnet_flag
        self.init_weight()
        if self.resnet_flag == "resnet50":
            self.backbone = ResNet50(pretrained=self.pretrained)
        elif self.resnet_flag == "resnet101":
            self.backbone = ResNet101(pretrained=self.pretrained)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


    def init_weight(self):
        if not self.pretrained:
            for m in self.modules():
                init_weight_backbone(m)
        else:
            init_weight_head(self.head)


    def forward(self,x):
        features = self.backbone(x)
        return self.head(self.decoder(features))



class CenterNet_ResNetX(nn.Module):
    def __init__(self,numclass,pretrained):
        super(CenterNet_ResNetX,self).__init__()
        self.backbone = ResNetn50(pretrained=pretrained)
        self.decoder = resnet_DecoderX(2048)
        self.head = resnet_Head(channel=64,num_classes=numclass)
        self.num_class = numclass
        self.pretrained = pretrained
        self.init_weight()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def init_weight(self):
        if not self.pretrained:
            for m in self.modules():
                init_weight_backbone(m)
        else:
            init_weight_head(self.head)


    def forward(self,x):
        features=self.backbone(x)
        return self.head(self.decoder(features))


class CenterNet_DLA34(nn.Module):
    def __init__(self,numclasses,pretrained):
        super(CenterNet_DLA34,self).__init__()
        self.pretrained = pretrained
        self.num_classes = numclasses
        self.heads = {'hm': self.num_classes,
                      'depth': 1,
                      'reg': 2}

        self.model = get_pose_net(num_layers=34, heads=self.heads)
        self.init_weight()

    def init_weight(self):
        if not self.pretrained:
            for m in self.modules():
                init_weight_backbone(m)
                
    def unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        feat = self.model(x)
        return feat[0]['hm'].sigmoid_(), feat[0]['depth'], feat[0]['reg']


