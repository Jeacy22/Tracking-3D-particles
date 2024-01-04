import colorsys
import os
import time
import numpy as np
import torch

from PIL import ImageDraw,ImageFont
import cv2

from models.model import CenterNet_Resnet,CenterNet_ConvNext,CenterNet_Dla34,CenterNet_ResnetX
from utils1.utils import cvtColor,get_classes,preprocess_input,resize_image,show_config

from utils1.utils_bbox import decode_bbox,postprocess,decode_bbox_pos



class CenterNet(object):
    _defaults = {
        'model_path':'N:\\3D_track\\Centernet_X\\logs\\resnet101\\best_epoch_weights.pth',
        'class_path':'database/classes.txt',
        'backbone':'resnet101',
        'input_shape':[1024,1024],
        'confidence':0.2,
        'nms_iou':0,
        'nms':True,
        'letterbox_image':False,
        'cuda' : False
    }

    @classmethod
    def get_defaults(cls,n):
        if n in cls.defaults():
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self,**kwargs):
        self.__dict__.update(self._defaults)
        for name,value in kwargs.items():
            setattr(self,name,value)
            self._defaults[name] = value

        self.class_names,self.num_classes = get_classes(self.class_path)

        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        #self.colors = list(map(lambda x: (int(255), int(0), int(0)), self.colors))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.generate()

        show_config(**self._defaults)

    def generate(self,onnx=False):
        assert self.backbone in ['resnet101',"resnet50X", 'dla34','dlax','convnext','swim_transformer']
        if self.backbone == "resnet101":
            self.net = CenterNet_Resnet(num_classes=self.num_classes, pretrained=False)
        elif self.backbone == "resnet50X":
            self.net = CenterNet_ResnetX(num_classes=self.num_classes, pretrained=False)

        elif self.backbone == 'dla34':
            self.net = CenterNet_Dla34(num_classes=self.num_classes)

        elif self.backbone == 'convnext':
            self.net = CenterNet_ConvNext(num_classes=self.num_classes, pretrained=False)



        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = torch.nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def detect_image(self,image,crop = False,count = False):
        image_shape = np.array(np.shape(image)[0:2])
        result=[]
        image = cvtColor(image)
        image_data,gw,gh = resize_image(image,(self.input_shape[1],self.input_shape[0]),self.letterbox_image)

        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)

            outputs = decode_bbox(outputs[0],outputs[1],outputs[2],gw,gh,self.confidence,self.cuda)
            results = postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, self.nms_iou)

            if results[0] is None:
                return image

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 5]
            top_boxes = results[0][:, :4]
            top_height = results[0][:,4]

        font1 = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(2e-2 * np.shape(image)[1]).astype('int32'))

        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0], 1)
        # ---------------------------------------------------------#
        #   计数
        # ---------------------------------------------------------#
        if count:
            print("top_label:", top_label)
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))

                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)

        for i,c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]
            depth = top_height[i]

            top1,left1,bottom1,right1 = box
            height1 = top_height[i]
            top = max(0, np.floor(top1).astype('int32'))
            left = max(0, np.floor(left1).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom1).astype('int32'))
            right = min(image.size[0], np.floor(right1).astype('int32'))
            center_y = (top1 + bottom1) / 2.
            center_x = (left1 + right1) / 2.
            label = '{} {:.2f}'.format(predicted_class, score)
            label_xy = '({:.2f}um)'.format(height1)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label_xy, font1)
            label = label.encode('utf-8')
            label_xy = label_xy.encode('utf-8')
            print(label, top, left, bottom, right)



            if top - label_size[1] >= 0:
                text_origin = np.array([center_x-60, center_y-160])
            else:
                text_origin = np.array([center_x-60, center_y-160])

            for i in range(thickness):
                #draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
                draw.arc([center_x-40,center_y-40,center_x+40,center_y+40], start=0, end=360,fill=self.colors[c],width=10)

            #draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c],width=4)
            draw.text(text_origin, str(label_xy, 'UTF-8',), fill=(255,255,255), font=font1)


            del draw

        return image

    def get_FPS(self, image, test_interval):

        image_shape = np.array(np.shape(image)[0:2])

        image = cvtColor(image)

        image_data,gw,gh = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()

            outputs = self.net(images)

            outputs = decode_bbox(outputs[0], outputs[1], outputs[2],gw,gh, self.confidence, self.cuda)

            results = postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, self.nms_iou)

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():

                outputs = self.net(images)

                outputs = decode_bbox(outputs[0], outputs[1], outputs[2],gw,gh, self.confidence, self.cuda)

                results = postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image,
                                      self.nms_iou)
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def detect_heatmap(self, image, heatmap_save_path,image_save_path):
        import cv2
        import matplotlib.pyplot as plt
        import os

        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


        image = cvtColor(image)

        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        image_data = np.expand_dims(np.transpose(preprocess_input(np.asarray(image_data[0], dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()

            outputs = self.net(images)
            cv2.imwrite(image_save_path, images[0].permute(1,2,0).cpu().numpy())



        score = np.max(outputs[0][0].permute(1, 2, 0).cpu().numpy(), -1)
        #score = cv2.resize(score, (image.size[0], image.size[1]))
        normed_score = (score * 255).astype('uint8')
        mask = np.zeros((normed_score.shape[1], normed_score.shape[0]))
        mask = np.maximum(mask, normed_score)
        cv2.imwrite(heatmap_save_path, mask)






    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w")

        image_shape = np.array(np.shape(image)[0:2])

        image = cvtColor(image)

        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)

            outputs = decode_bbox(outputs[0], outputs[1], outputs[2], self.confidence, self.cuda)

            results = postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, self.nms_iou)

            if results[0] is None:
                return

            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box

            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (
            predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return

    def detect_image2(self,image):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data,gw,gh = resize_image(image,(self.input_shape[1],self.input_shape[0]),self.letterbox_image)

        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)

            outputs = decode_bbox(outputs[0],outputs[1],outputs[2],gw,gh,self.confidence,self.cuda)
            results = postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, self.nms_iou)

            if results[0] is None:
                return image,0,0,0

            top_label = np.array(results[0][:8, 6], dtype='int32')
            top_conf = results[0][:8, 5]
            top_boxes = results[0][:8, :4]
            top_depth = results[0][:8, 4]


            len1=len( top_label)


        return top_boxes,top_depth,top_label,len1
