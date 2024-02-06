from torch.utils.data import Dataset
import numpy as np
import os
import torch
import json
from PIL import Image
from lxml import etree
import cv2

class MyDataSet(Dataset):
    """
    读取自己的数据集
    一个用于数据读取的函数必须实现两个方法
    __len__：返回数据集的长度
    __getitem__:返回一张图片及其标注信息
    """
    def __init__(self,data_root,transforms=None,img_set="train.txt"):
        """
        :param data_root: 数据集的路径
        :param transforms: 对图像进行预处理，比如说翻转，裁剪等等，数据增强的一种方法
        :param img_set: 判断到底是训练集，验证集
        :return:
        """
        self.imgroot=os.path.join(data_root,"PNGimage")
        #图片路径
        self.annotations=os.path.join(data_root,"Annotation")
        #标注路径
        self.txt_path=os.path.join(data_root,"Imageset","Main",img_set)
        #标注框之类的信息
        with open(self.txt_path) as read:
            xml_list=[os.path.join(self.annotations,line.strip()+".xml")
                      for line in read.readlines() if len(line.strip())>0]
            #读取对应的xml文件

        self.xml_list=[]
        #下面是判断xml文件中是否存在不满足条件的
        for xml_path in xml_list:
            if os.path.exists(xml_path) is False:
                print("Warning: not found '{xml_path}', skip this annotation file.")
                continue
            with open(xml_path) as fid:
                xml_str=fid.read()
            xml=etree.fromstring(xml_str)
            data=self.parse_xml_to_dict(xml)["annotation"]
            #读取xml里面的一些数据和特征
            if "object" not in data:
                print(f"INFO: no objects in {xml_path}, skip this annotation file.")
                continue
            self.xml_list.append(xml_path)
        assert  len(self.xml_list)>0,"in '{}' file does not find any information.".format(self.txt_path)

        #read class_indict
        #print(os.getcwd()) #获取当前路径
        json_file='database/classes.json'
        assert os.path.exists(json_file), "{} file not exist.".format(json_file)
        with open(json_file) as f:
            self.class_dict=json.load(f)

        self.transforms=transforms
    def __len__(self):
        return len(self.xml_list)
    #返回读取到的数据集的长度

    def __getitem__(self,idx):
        #根据idx返回对应的图片和相关标注信息
        xml_path=self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str=fid.read()
        xml=etree.fromstring(xml_str)
        data=self.parse_xml_to_dict(xml)["annotation"]
        img_path = os.path.join(self.imgroot,data["filename"])
        image =cv2.imread(img_path)

        boxes = []
        labels = []
        iscrowd = []

        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            depth = float(obj["bndbox"]["deptho"])/40.
            #有的标准信息中会出现w或h为0的情况，这样的数据会导致计算回归loss变为nan
            if xmax<=xmin or ymax<=ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                continue
            boxes.append([xmin,ymin,xmax,ymax,depth])
            labels.append(self.class_dict[obj["name"]])
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)
        #convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        #truth ground
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image,target = self.transforms(image,target)

        return image, target

    def get_height_and_width(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        """
        将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree

        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def data_index(self,idx):
        """
        该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
        由于不用去读取图片，可大幅缩减统计时间
        :param idx:输入需要获取图像的索引
        :return:
        """
        xml_path=self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml=etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        bboxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            depth = float(obj["bndbox"]["deptho"])/40.
            bboxes.append([xmin,ymin,xmax,ymax,depth])
            labels.append(self.class_dict[obj["name"]])
            iscrowd.append(int(obj["difficult"]))

        boxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return (data_height,data_width),target

    def coco_index(self, idx):
        """
        该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
        由于不用去读取图片，可大幅缩减统计时间

        Args:
            idx: 输入需要获取图像的索引
        """
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        # img_path = os.path.join(self.img_root, data["filename"])
        # image = Image.open(img_path)
        # if image.format != "JPEG":
        #     raise ValueError("Image format not JPEG")
        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            depth = float(obj["bndbox"]["deptho"])/40.
            boxes.append([xmin, ymin, xmax, ymax, depth])
            labels.append(self.class_dict[obj["name"]])
            iscrowd.append(int(obj["difficult"]))

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return (data_height, data_width), target


    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))




"""
import transforms
from draw_box_utils import draw_objs
from PIL import Image
import json
import matplotlib.pyplot as plt
import random
import torchvision.transforms as ts
from network import transform
from train_utils import distribution_utils as utils


def tensor_numpy(clean):      # 去掉batch通道 (batch, C, H, W) --> (C, H, W)
    clean = np.around(clean.mul(255))                     # 转换到颜色255 [0, 1] --> [0, 255]
    clean = np.uint8(clean).transpose(1, 2, 0)            # 跟换三通道 (C, H, W) --> (H, W, C)
    r, g, b = cv2.split(clean)                             # RGB 通道转换
    clean = cv2.merge([b, g, r])
    return clean

category_index={}
try:
    json_file=open('./classes.json',"r")
    class_dict = json.load(json_file)
    category_index = {str(v):str(k) for k,v in class_dict.items()}
except Exception as e:
    print(e)
    exit(-1)

data_transform = {
        "train": transforms.Compose([transforms.AddGaussianNoise(),transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }
data_root="L:/3D_track/data_big/data4"
train_dataset=MyDataSet(data_root,data_transform["train"],"train11.txt")
val_data_set_loader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      collate_fn=train_dataset.collate_fn)

print(len(train_dataset))

metric_logger = utils.MetricLogger(delimiter="")
print_freq=50
header="Train"

for images, targets in metric_logger.log_every(val_data_set_loader, print_freq, header):
    for m in range(len(images)):
        image = images[m]
        image = tensor_numpy(image)
        cv2.imshow('transformers', image)

        cv2.waitKey(0)



#随机抽取图片显示

font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
for index in random.sample(range(0,len(train_dataset)),k=6):
    image,target=train_dataset[index]
    image_std = [0.049, 0.049, 0.049]
    image_mean = [0.49, 0.49, 0.49]
    #transform = transform.GeneralizedRCNNTransform(1000, 1000, image_mean, image_std)
    image=tensor_numpy(image)
    boxes= target["boxes"].numpy()
    classes=target["labels"].numpy()
    for box, cls in zip(boxes, classes):
        xmin, ymin, xmax, ymax,depth = box
        centerx = (xmin+xmax)/2
        centery =(ymin+ymax)/2
        if int(cls)==1:
            color=[255,0,0]
        elif int(cls)==2:
            color=[0,255,0]
        image = cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
        cv2.putText(image, '{:.3f}'.format(depth), (int(centerx) - 30, int(ymin) - 15), font, 1, color, 4)
        cv2.circle(image, (int(centerx), int(centery)), 3, color, 3)
    cv2.imshow('gt', image)

    cv2.waitKey(0)
"""












