import argparse
import os
from pathlib import Path
import cv2
from CenterNetX_detection import detect_single_image,detect_heatmap,detect_fps
from model.models import CenterNet_ResNet,CenterNet_ResNetX,CenterNet_DLA34
import torch
from utils import get_config
from PIL import Image

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--model_name', type=str,default='CenterNet_ResNetX',help="the name of trained model")
    parse.add_argument("--weigths", type=str,default=r"/root/autodl-tmp/Tracking-3D-particles/weight_saved/weigth.pth",help="the path to weigths")
    parse.add_argument("--input_size",type=int,default=[1024,1024],help="the initial image size of the model")
    parse.add_argument("--mode",type=str,default="single image prediction",help="get fps, single image prediction, get heatmap, imageset ")
    parse.add_argument("--cuda",default=True,help="whether to use cuda")
    parse.add_argument("--result_image_save_path", type=str, default="",
                       help="the path to save the result,if ypu dont "
                            "want to save the result, you can not input the parameter")

    ## the config of single image prediction
    parse.add_argument("--nms_threshold",type=float,default=0.3,help="the threshold to NMS algorithm")
    parse.add_argument("--confidence_threshold",type=float,default=0.3,help="the threshold to confidence")
    parse.add_argument("--box_length",type=int,default=40,help="the length of the box's side")

    ##the config of image dataset
    parse.add_argument("--data_file_path",type=str,default="",help="the path to imageset")
    parse.add_argument("--save_file_path", type=str, default="", help="the path to save the predictions of imageset")


    args = parse.parse_args()

    return args

def singal_image_prediction(input_size,model,nms_threshold,confidence_threshold,cuda,box_length,save_path,num_classes,classes):
    while True:
        image_file = input('Input image filename:')
        try:
            image = Image.open(image_file)
        except:
            print("Error! Can't open image! Try again!")
        else:
            outputs,infos = detect_single_image(image,input_size,model,nms_threshold,confidence_threshold,cuda,box_length,
                                          letterbox_image=True,count=True,num_classes=num_classes,classes = classes)
            if os.path.exists(save_path):
                outputs.save(os.path.join(save_path,"result.png"))

            outputs.show()
def get_heatmap(model,input_size,cuda,save_path):
    while True:
        image_file = input()
        try:
            image = cv2.imread(image_file)
        except:
            print("Error! Can't open image! Try again!")
        else:
            results = detect_heatmap(image,model,input_size,cuda)

            if os.path.exists(save_path):
                results.save(os.path.join(save_path,"heatmap.png"))

            results.show()
def get_fps(model,input_size,cuda,test_interval,confidence_threshold,nms_threshold,box_length,letterbox_image):
    while True:
        image_file = input()
        try:
            image = cv2.imread(image_file)
        except:
            print("Error! Can't open image! Try again!")
        else:
            result = detect_fps(image,model,input_size,cuda,test_interval,confidence_threshold,nms_threshold,box_length,letterbox_image)
            print(str(result) + ' seconds, ' + str(1 / result) + 'FPS, @batch_size 1')


def image_dataset_prediction(data_file,model,input_size,cuda,nms_threshold,confidence_threshold,box_length,save_path,num_classes,classes):
   if os.path.exists(data_file):
       for dirpath,subdir,files in os.walk(data_file):
           for file in files:
               image = cv2.imread(str(os.path.join(dirpath,file)))
               image_result,infos = detect_single_image(image, input_size, model, nms_threshold, confidence_threshold, cuda,
                                             box_length,
                                             letterbox_image=True, count=True, num_classes=num_classes,
                                             classes=classes)
               with open("dataset.txt", "a+") as f:
                    f.writelines(['  '.join(infos)+'\n'])
               if os.path.exists(save_path):
                   image_result.save(os.path.join(save_path,file))
       print("Finished processing")

   else:
       print("Error! Can't open the file! Try again!")




if __name__ == "__main__":
    args = parse_args()
    current_path = Path.cwd()
    config_dict = get_config(current_path)
    classes = config_dict['classes'].split(',')
    num_classes = len(classes)

    if args.model_name == "CenterNet_ResNet50":
        model = CenterNet_ResNet(numclass=num_classes,pretrained=False,resnet_flag="resnet50")
    if args.model_name == "CenterNet_ResNetX":
        model = CenterNet_ResNetX(numclass=num_classes,pretrained=False)
    if args.model_name == "CenterNet_DLA34":
        model = CenterNet_DLA34(numclasses=num_classes,pretrained=False)
    if args.model_name == "CenterNet_ResNet101":
        model = CenterNet_ResNet(numclass=num_classes, pretrained=False, resnet_flag="resnet101")

    if args.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device=device)
    model.load_state_dict(torch.load(args.weigths,map_location=device))
    model.eval()


    if args.mode == "single image prediction":
        singal_image_prediction(args.input_size,model,args.nms_threshold,
                                         args.confidence_threshold,args.cuda,args.box_length,args.result_image_save_path,num_classes,classes)
    if args.mode == "get fps":
        get_fps(model,args.input_size,args.cuda,args.test_interval,args.confidence_threshold,
                args.nms_threshold,args.box_length,args.letterbox_image)

    if args.mode == 'get heatmap':
        get_heatmap(model,args.input_size,args.cuda,args.result_image_save_path)

    if args.mode == 'imageset prediction':
        image_dataset_prediction(args.data_file,model,args.input_size,args.cuda,args.nms_threshold,
                                 args.confidence_threshold,args.box_length,args.save_path,num_classes,classes)





