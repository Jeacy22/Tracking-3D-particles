
import torch.nn as nn
import torch
from torch.hub import load_state_dict_from_url
import math

from .ConvNext import Multi_Concat_Block,SiLU,Conv,SPPCSPC
model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}



class BasicBlock(nn.Module):
    expansion = 1 #每一个conv的卷积核个数的倍数

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):#downsample对应虚线残差结构
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)#BN处理
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x #捷径上的输出值
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

#50,101,152
class Bottleneck(nn.Module):
    expansion = 4#4倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion,#输出*4
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):#block残差结构 include_top为了之后搭建更加复杂的网络
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)自适应
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x4,x3,x2


def resnet34(include_top=True,):
    return ResNet(BasicBlock, [3, 4, 6, 3], include_top=include_top)




def resnet101_x( include_top=True,pretrained = True):
        model=ResNet(Bottleneck, [3, 4, 23, 3], include_top=include_top)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['resnet101'], model_dir='pretrained_train/')
            model.load_state_dict(state_dict)

        # extract feature

        return model


def resnet50_x(include_top=True, pretrained=True):
    model = ResNet(Bottleneck, [3, 4, 6, 3], include_top=include_top)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet50'], model_dir='pretrained_train/')
        model.load_state_dict(state_dict)

    # extract feature

    return model


class resnet_Decoder(nn.Module):
    def __init__(self,inplanes,bn_momentum=0.1):
        super(resnet_Decoder,self).__init__()
        self.bn_momentum = bn_momentum
        self.inplanes = inplanes
        self.cardinality = 8
        self.deconv_with_bias = False
        self.deconv_layers = self._make_deconv_layer(
            num_layers=3,
            num_filters=[256, 128, 64],
            num_kernels=[4, 4, 4],
        )


    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            kernel = num_kernels[i]
            planes = num_filters[i]

            layers.append(nn.ConvTranspose2d(in_channels=self.inplanes,out_channels=planes,kernel_size=kernel,
                    stride=2,padding=1,output_padding=0,bias=self.deconv_with_bias))

            layers.append(nn.BatchNorm2d(planes, momentum=self.bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
        return nn.Sequential(*layers)



    def forward(self, x):
        return self.deconv_layers(x)

class resnet_Head(nn.Module):
    def __init__(self, num_classes=80, channel=64, bn_momentum=0.1):
        super(resnet_Head, self).__init__()

        self.cls_head = nn.Sequential(
            nn.Conv2d(64, channel,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, num_classes,
                      kernel_size=1, stride=1, padding=0))
        # 宽高预测的部分
        self.depth_head = nn.Sequential(
            nn.Conv2d(64, channel,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1,
                      kernel_size=1, stride=1, padding=0))

        # 中心点预测的部分
        self.reg_head = nn.Sequential(
            nn.Conv2d(64, channel,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 2,
                      kernel_size=1, stride=1, padding=0))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        hm = self.cls_head(x).sigmoid_()
        depth = self.depth_head(x)
        offset = self.reg_head(x)
        return hm,depth, offset


class resnet_DecoderX(nn.Module):
    def __init__(self,inplanes,bn_momentum=0.1):
        super(resnet_DecoderX,self).__init__()
        self.bn_momentum = bn_momentum
        self.inplanes = inplanes
        self.deconv_with_bias = False
        self.basic_inchannel = 512
        self.basic_outchannel = 64

        self.sppcspc1  = SPPCSPC(self.basic_inchannel*4, self.basic_outchannel*4)
        self.deconv_layer1 =  nn.ConvTranspose2d(
            in_channels=self.basic_outchannel*4,out_channels=self.basic_outchannel*4,kernel_size=4,
            stride=2,padding=1,output_padding=0,bias=self.deconv_with_bias)


        self.sppcspc2 = SPPCSPC(self.basic_inchannel*2, self.basic_outchannel*4)
        self.deconv_layer2 = nn.ConvTranspose2d(
            in_channels=self.basic_inchannel*2, out_channels=self.basic_outchannel * 2, kernel_size=4,
            stride=2, padding=1, output_padding=0, bias=self.deconv_with_bias)


        self.sppcspc3 = SPPCSPC(self.basic_inchannel, self.basic_outchannel*2)
        self.deconv_layer3 = nn.ConvTranspose2d(
            in_channels=self.basic_inchannel , out_channels=self.basic_outchannel, kernel_size=4,
            stride=2, padding=1, output_padding=0, bias=self.deconv_with_bias)


        self.deconv_layer4 =  nn.ConvTranspose2d(
            in_channels=self.basic_outchannel*4, out_channels=self.basic_outchannel*2, kernel_size=4,
            stride=2, padding=1, output_padding=0, bias=self.deconv_with_bias)


        self.deconv_layer5 = nn.ConvTranspose2d(
            in_channels=self.basic_outchannel * 2, out_channels=self.basic_outchannel, kernel_size=4,
            stride=2, padding=1, output_padding=0, bias=self.deconv_with_bias)

        self.contact_block1 = Multi_Concat_Block(self.basic_outchannel*4, self.basic_outchannel*8,
                                                 self.basic_outchannel*4)

        self.contact_block2 = Multi_Concat_Block(self.basic_outchannel*2, self.basic_outchannel*4,
                                                 self.basic_outchannel*2)

        self.contact_block3 = Multi_Concat_Block(self.basic_outchannel, self.basic_outchannel * 2,
                                                 self.basic_outchannel)



        self.apply(self._init_weights)


    def _init_weights(self, m):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        x3, x2, x1 = x
        x3 = self.sppcspc1(x3)
        x3 = self.deconv_layer1(x3)

        x = self.sppcspc2(x2)

        x3 = x3 + x

        x3 = self.contact_block1(x3)

        x3 = self.deconv_layer4(x3)
        x2 = self.deconv_layer2(x2)

        x3 = x3 + x2

        x = self.sppcspc3(x1)

        x3 = x3 + x

        x3 = self.contact_block2(x3)

        x3 = self.deconv_layer5(x3)
        x1 = self.deconv_layer3(x1)

        x3 = x3 + x1

        x3 = self.contact_block3(x3)
        return x3
























