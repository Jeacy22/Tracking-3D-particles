import numpy as np
from PIL import Image
from PIL import ImageFont,ImageDraw
import colorsys
from dataset.dataloader import preprocess_input
import torch
from utils_bbox import decode_bbox,postprocess
import cv2
import time

def reshape_image(image,outputsize,letterbox_image,box_length):
    iw, ih = image.size
    ow, oh = outputsize
    scale = min(ow / iw, oh / ih)
    gw = float(box_length * scale)
    gh = float(box_length * scale)
    if letterbox_image:

        nw = int(scale * iw)
        nh = int(scale * ih)

        image = image.resize((nw,nh),Image.BICUBIC)
        image_new = Image.new('RGB',outputsize,(128,128,128))
        image_new.paste(image, ((ow - nw) // 2, (oh - nh) // 2))
    else:
        image_new = image.resize((ow, oh), Image.BICUBIC)
    return image_new, gw, gh


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

def detect_single_image(image,input_size,model,nms_threshold,confidence_threshold,cuda,box_length,
                        letterbox_image,count,num_classes,classes,show_flag=True):
    infos = list()
    image_shape = np.array((np.shape(image))[0:2])
    image = cvtColor(image)
    image_data,gw,gh = reshape_image(image,input_size,letterbox_image=letterbox_image,box_length=box_length)
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
    with torch.no_grad():
        images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
        if cuda:
            images = images.cuda()

        output = model(images)
        outputs = decode_bbox(output[0],output[1],output[2],gw,gh,confidence_threshold,cuda)
        results = postprocess(outputs,image_shape,input_size,nms_threshold)

        
        if results[0] is None:
            return image

        top_label = np.array(results[0][:, 6], dtype='int32')
        top_conf = results[0][:, 5]
        top_boxes = results[0][:, :4]
        top_depth = results[0][:, 4]

        if count:
            class_num = np.zeros([num_classes], dtype='int32')
            for i in range(num_classes):
                num = np.sum(top_label == i)
                class_num[i] = num

        for i,c in enumerate(top_label):
            predicted_class = classes[int(c)]
            box = top_boxes[i]
            score = top_conf[i]
            depth = top_depth[i]

            top,left,bottom,right = box
            centery = (top+bottom)/2.
            centerx = (left+right)/2.
            if count:
                info = [int(c), score, centerx, centery, depth,class_num]
            else:
                info = [int(c), score, centerx, centery, depth]
            infos.append(info)

            """
            top =  max(0,np.floor(top).astype(int))
            left = max(0,np.floor(left).astype(int))
            bottom = min(image.size[1],np.floor(bottom).astype(int))
            right = min(image.size[0],np.floor(right).astype(int))
            """
            if show_flag:
                font  = ImageFont.truetype(font = 'arial.ttf',size = np.floor(2e-2*np.shape(image)[1]).astype(int))

                label = "{}: {:.2f}".format(predicted_class, score)
                label_depth = "{:.2f}um".format(depth)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font=font)
                label = label.encode("utf-8")
                label_depth = label_depth.encode("utf-8")
                print(label,centerx,centery,label_depth)
                hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
                colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
                colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),colors))

                text_origin = np.array([centerx-60,centery-110])
                draw.arc([centerx - 40, centery - 40, centerx + 40, centery + 40], start=0, end=360,
                         fill=colors[c], width=10)

                draw.text(text_origin,str(label_depth,"UTF-8"),font=font, fill=(255,255,255))

                del draw

        return image,infos



def detect_heatmap(image,model,input_size,cuda):
    image = cv2.cvtColor(image)
    ow,oh = image.shape
    tw,th = input_size
    image = image.resize((tw, th), Image.BICUBIC)
    image_data = np.expand_dims((np.transpose(preprocess_input(image), (2, 0, 1))), 0)
    with torch.no_grad():
        images = torch.from_numpy(image_data).dtype(torch.FloatTensor)
        if cuda:
            images = images.cuda()

        output = model(images)

        scores = np.max(output[0][0].permute(2,0,1).cpu().numpy(), axis=0)

        scores = scores.resize((ow,oh), Image.BICUBIC)
        new_scores = (scores*255).astype(np.uint8)
        mask = np.zeros_like(new_scores)
        heatmap = np.maximum(mask,new_scores)
        return heatmap



def detect_fps(image,model,input_size,cuda,test_interval,confidence_threshold,nms_threshold,box_length,letterbox_image):
    image_shape = np.array((np.shape(image))[0:2])
    image = cv2.cvtColor(image)
    image, gw, gh = reshape_image(image, input_size, letterbox_image=letterbox_image, box_length=box_length)
    image_data = np.expand_dims((np.transpose(preprocess_input(image), (2, 0, 1))), 0)
    with torch.no_grad():
        images = torch.from_numpy(image_data).dtype(torch.FloatTensor)
        if torch.cuda:
            images = images.cuda()

        output = model(images)
        outputs = decode_bbox(output[0], output[1], output[2], gw, gh, confidence_threshold, cuda)
        results = postprocess(outputs, image_shape, input_size, nms_threshold)

        t1 = time.time()
        for _ in range(test_interval):
            output = model(images)
            outputs = decode_bbox(output[0], output[1], output[2], gw, gh, confidence_threshold, cuda)
            results = postprocess(outputs, image_shape, input_size, nms_threshold)
        t2=time.time()

        time_space = (t2-t1) * test_interval
        return time_space









































