import math
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image
from utils1.utils import cvtColor,preprocess_input
import cv2
from copy import deepcopy

def draw_guassian(heatmap,center,radius,k=1):
    diameter = 2*radius+1
    gaussian = guassian2D((diameter,diameter),sigma = diameter/6)
    x,y = int(center[0]),int(center[1])
    height,width = heatmap.shape[0:2]

    left,right = min(x,radius),min(width-x,radius+1)
    top,bottom = min(y,radius),min(height-y,radius+1)

    masked_heatmap = heatmap[y-top:y+bottom,x-left:x+right]
    masked_gaussian = gaussian[radius-top:radius+bottom,radius-left:radius+right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap



def guassian2D(shape,sigma=1):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def guassian_radius(det_szie,min_overlap=0.9):
    height,width = det_szie

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


class CenternetDataset(Dataset):
    def __init__(self,annotation_lines,input_shape,num_classes,train):
        super(CenternetDataset,self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(self.annotation_lines)

        self.input_shape = input_shape
        self.output_shape = (int(input_shape[0]/4),int(input_shape[1]/4))
        self.num_classes = num_classes
        self.train = train

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        image,box = self.get_random_data(self.annotation_lines[index],self.input_shape,random = self.train)
        batch_hm = np.zeros((self.output_shape[0], self.output_shape[1], self.num_classes), dtype=np.float32)
        batch_depth = np.zeros((self.output_shape[0], self.output_shape[1], 1), dtype=np.float32)
        batch_reg = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        batch_reg_mask = np.zeros((self.output_shape[0], self.output_shape[1]), dtype=np.float32)

        if len(box) !=0:
            boxes = np.array(box[:,:4],dtype=np.float32)
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] / self.input_shape[1] * self.output_shape[1], 0,
                                       self.output_shape[1] - 1)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] / self.input_shape[0] * self.output_shape[0], 0,
                                       self.output_shape[0] - 1)

        for i in range(len(box)):
            bbox = deepcopy(boxes[i])
            cls_id = int(box[i,-1])
            h,w = bbox[3]-bbox[1],bbox[2]-bbox[0]
            if h>0 and w>0:
                radius = guassian_radius((math.ceil(h),math.ceil(w)))
                radius = max(0,int(radius))
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                batch_hm[:,:,cls_id] = draw_guassian(batch_hm[:,:,cls_id],ct_int,radius)
                batch_depth[ct_int[1],ct_int[0]]=box[i,4]
                batch_reg_mask[ct_int[1],ct_int[0]]=1
                batch_reg[ct_int[1], ct_int[0]] = ct - ct_int

        image = np.transpose(preprocess_input(image), (2, 0, 1))
        #image = preprocess_input(image)

        return image,batch_hm, batch_depth, batch_reg, batch_reg_mask


    def rand(self,a=0,b=1):
        return np.random.rand()*(b-a)+a


    def get_random_data(self,annotation_line,input_shape,jitter=.3,hue=.1,sat=0.7,val=0.4,random=True):
         line = annotation_line.split()
         image = Image.open(line[0])
         image = cvtColor(image)

         iw,ih = image.size
         h,w =input_shape
         box = np.array([np.array(list(map(float,box.split(',')))) for box in line[1:]])

         if not random:
             scale = min(w/iw,h/ih)
             nw = int(iw*scale)
             nh = int(ih*scale)
             dx = (w-nw)//2
             dy = (h-nh)//2

             image = image.resize((nw,nh),Image.BICUBIC)
             new_image = Image.new('RGB',(w,h),(128,128,128))
             new_image.paste(image,(dx,dy))
             image_data = np.array(new_image,np.float32)

             if  len(box)>0:
                 np.random.shuffle(box)
                 box[:,[0,2]] = box[:,[0,2]]*nw/iw+dx
                 box[:,[1,3]] = box[:,[1,3]]*nh/ih+dy
                 pos = np.where(box[:, 0:2] < 0)
                 pos = np.array(list(pos)).astype(dtype=int).tolist()
                 box = np.delete(box, pos[0], axis=0)

                 pos = np.where(box[:, 2] > w)
                 pos = np.array(list(pos)).astype(dtype=int).tolist()
                 box = np.delete(box, pos[0], axis=0)

                 pos = np.where(box[:, 3] > h)
                 pos = np.array(list(pos)).astype(dtype=int).tolist()
                 box = np.delete(box, pos[0], axis=0)

                 box_w = box[:, 2] - box[:, 0]
                 box_h = box[:, 3] - box[:, 1]
                 box[:, 4] = box[:, 4] / 50
                 box = box[np.logical_and(box_w > 1, box_h > 1)]
             return image_data,box


         new_ar =w/h*self.rand(1-jitter,1+jitter)/self.rand(1-jitter,1+jitter)
         scale = self.rand(.25,2)

         if new_ar<1:
             nh = int(scale*h)
             nw = int(nh*new_ar)
         else:
             nw = int(scale*w)
             nh = int(nw/new_ar)

         image = image.resize((nw,nh),Image.BICUBIC)
         dx = int(self.rand(0,w-nw))
         dy = int(self.rand(0,h-nh))

         new_image = Image.new('RGB',(w,h),(128,128,128))
         new_image.paste(image,(dx,dy))
         image = new_image

         flip = self.rand()<.5
         if flip:
             image = image.transpose(Image.FLIP_LEFT_RIGHT)

         image_data = np.array(image,np.uint8)

         r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1

         hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
         dtype = image_data.dtype

         x = np.arange(0, 256, dtype=r.dtype)
         lut_hue = ((x * r[0]) % 180).astype(dtype)
         lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
         lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

         image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
         image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

         if len(box) > 0:
             np.random.shuffle(box)
             box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
             box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
             if flip:
                 box[:, [0, 2]] = w - box[:, [2, 0]]
             pos = np.where(box[:, 0:2] < 0)
             pos = np.array(list(pos)).astype(dtype=int).tolist()
             box = np.delete(box, pos[0], axis=0)

             pos = np.where(box[:, 2] > w)
             pos = np.array(list(pos)).astype(dtype=int).tolist()
             box = np.delete(box, pos[0], axis=0)

             pos = np.where(box[:, 3] > h)
             pos = np.array(list(pos)).astype(dtype=int).tolist()
             box = np.delete(box, pos[0], axis=0)


             box_w = box[:, 2] - box[:, 0]
             box_h = box[:, 3] - box[:, 1]
             box[:, 4] = box[:, 4] / 50
             box = box[np.logical_and(box_w > 1, box_h > 1)]

         return image_data,box

def centernet_dataset_collate(batch):
    imgs, batch_hms, batch_whs, batch_regs, batch_reg_masks = [], [], [], [], []

    for img, batch_hm, batch_wh, batch_reg, batch_reg_mask in batch:
        imgs.append(img)
        batch_hms.append(batch_hm)
        batch_whs.append(batch_wh)
        batch_regs.append(batch_reg)
        batch_reg_masks.append(batch_reg_mask)

    imgs = torch.from_numpy(np.array(imgs)).type(torch.FloatTensor)
    batch_hms = torch.from_numpy(np.array(batch_hms)).type(torch.FloatTensor)
    batch_whs = torch.from_numpy(np.array(batch_whs)).type(torch.FloatTensor)
    batch_regs = torch.from_numpy(np.array(batch_regs)).type(torch.FloatTensor)
    batch_reg_masks = torch.from_numpy(np.array(batch_reg_masks)).type(torch.FloatTensor)
    return imgs, batch_hms, batch_whs, batch_regs, batch_reg_masks

'''
if __name__=="__main__":
    from utils1.utils import get_classes
    from torch.utils.data import DataLoader
    batch_size=2
    num_workers = 1
    train_sampler = None
    shuffle = True
    classes_path = 'classes.txt'
    train_annotation_path = 'train.txt'
    class_names, num_classes = get_classes(classes_path)
    with open(train_annotation_path) as f:
        train_lines = f.readlines()

    num_train = len(train_lines)
    input_shape = [512, 512]

    train_dataset = CenternetDataset(train_lines,input_shape,num_classes,train=True)
    gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                     drop_last=True, collate_fn=centernet_dataset_collate, sampler=train_sampler)
    for i in range(num_train):
        i = i % num_train
        image_data, gt,batch_hm, batch_depth, batch_reg, batch_reg_mask=train_dataset.__getitem__(i)

        # to BGR
        image = image_data[..., (2, 1, 0)]*255
        # denormalize
        # to
        image = image.astype(np.uint8).copy()

        font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
        for box in gt:
            xmin, ymin, xmax, ymax, depth,cls = box
            centerx =(xmin+xmax)/2
            centery = (ymin+ymax)/2
            image = cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
            cv2.putText(image, '{:.3f}'.format(depth), (int(centerx) - 30, int(ymin) - 15), font, 1, (0, 255, 255), 4)
            if cls ==1:
                color=[0,0,255]
            else:
                color=[255,0,0]
            cv2.circle(image, (int(centerx), int(centery)), 3, color, 3)
        cv2.imshow('gt', image)

        cv2.waitKey(0)
'''







































