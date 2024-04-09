import torch.nn as nn
from model.RESNET import BasicBlock,Bottleneck,make_save_dir,loda_dict
from model.block import SPPCSPC,Multi_Concat_Block,Conv


class ResNetn(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        super(ResNetn, self).__init__()
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
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
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

def ResNetn34(pretrained):
    backbone = ResNetn(BasicBlock,[3,4,6,3],include_top=True)
    if pretrained and make_save_dir():
        backbone=loda_dict("resnet34",backbone)
    return backbone

def ResNetn50(pretrained):
    backbone = ResNetn(Bottleneck,[3,4,6,3],include_top=True)
    if pretrained and make_save_dir():
        backbone = loda_dict('resnet50',backbone)
    return backbone

def ResNetn101(pretrained):
    backbone = ResNetn(Bottleneck,[3,4,23,3],include_top=True)
    if pretrained and make_save_dir():
        backbone = loda_dict('resnet101',backbone)
    return backbone


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

        self.depth_head = nn.Sequential(
            nn.Conv2d(64, channel,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1,
                      kernel_size=1, stride=1, padding=0))


        self.reg_head = nn.Sequential(
            nn.Conv2d(64, channel,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 2,
                      kernel_size=1, stride=1, padding=0))


    def forward(self, x):
        hm = self.cls_head(x).sigmoid_()
        depth = self.depth_head(x)
        offset = self.reg_head(x)
        return hm,depth, offset


class resnet_DecoderX(nn.Module):
    def __init__(self,inplanes,bn_momentum=0.1,only_contact_block=False, only_SPPCSPC=False):
        super(resnet_DecoderX,self).__init__()
        self.bn_momentum = bn_momentum
        self.inplanes = inplanes
        self.deconv_with_bias = False
        self.basic_inchannel = 512
        self.basic_outchannel = 64
        self.only_contact_block = only_contact_block
        self.only_SPPCSPC = only_SPPCSPC

        if self.only_contact_block:
            self.sppcspc1 = Conv(self.basic_inchannel*4, self.basic_outchannel*4)
            self.sppcspc2 = Conv(self.basic_inchannel * 2, self.basic_outchannel * 4)
            self.sppcspc3 = Conv(self.basic_inchannel , self.basic_outchannel * 2)

        else:
            self.sppcspc1 = SPPCSPC(self.basic_inchannel * 4, self.basic_outchannel * 4)
            self.sppcspc2 = SPPCSPC(self.basic_inchannel * 2, self.basic_outchannel * 4)
            self.sppcspc3 = SPPCSPC(self.basic_inchannel, self.basic_outchannel * 2)

        if self.only_SPPCSPC:
            self.conv1 = Conv(self.basic_outchannel * 4, self.basic_outchannel * 4)
            self.conv2 = Conv(self.basic_outchannel * 2, self.basic_outchannel * 2)
            self.conv3 = Conv(self.basic_outchannel, self.basic_outchannel)
        else:

            self.contact_block1 = Multi_Concat_Block(self.basic_outchannel * 4, self.basic_outchannel * 8,
                                                     self.basic_outchannel * 4)
            self.contact_block2 = Multi_Concat_Block(self.basic_outchannel * 2, self.basic_outchannel * 4,
                                                     self.basic_outchannel * 2)
            self.contact_block3 = Multi_Concat_Block(self.basic_outchannel, self.basic_outchannel * 2,
                                                     self.basic_outchannel)


        self.deconv_layer1 =  nn.ConvTranspose2d(
            in_channels=self.basic_outchannel*4,out_channels=self.basic_outchannel*4,kernel_size=4,
            stride=2,padding=1,output_padding=0,bias=self.deconv_with_bias)

        self.deconv_layer2 = nn.ConvTranspose2d(
            in_channels=self.basic_inchannel*2, out_channels=self.basic_outchannel * 2, kernel_size=4,
            stride=2, padding=1, output_padding=0, bias=self.deconv_with_bias)

        self.deconv_layer3 = nn.ConvTranspose2d(
            in_channels=self.basic_inchannel , out_channels=self.basic_outchannel, kernel_size=4,
            stride=2, padding=1, output_padding=0, bias=self.deconv_with_bias)

        self.deconv_layer4 =  nn.ConvTranspose2d(
            in_channels=self.basic_outchannel*4, out_channels=self.basic_outchannel*2, kernel_size=4,
            stride=2, padding=1, output_padding=0, bias=self.deconv_with_bias)

        self.deconv_layer5 = nn.ConvTranspose2d(
            in_channels=self.basic_outchannel * 2, out_channels=self.basic_outchannel, kernel_size=4,
            stride=2, padding=1, output_padding=0, bias=self.deconv_with_bias)


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

