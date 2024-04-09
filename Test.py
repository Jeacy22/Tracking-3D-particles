import argparse
import os
import cv2
from model.models import CenterNet_ResNet,CenterNet_ResNetX,CenterNet_DLA34
from CenterNetX_detection import detect_single_image
import torch
from utils import get_config
from pathlib import Path
from PIL import Image
import numpy as np


def parse():
    parse = argparse.ArgumentParser(description="this is the test script")
    parse.add_argument("--test_annotation_path",type=str,default='dataset/test.txt',help="the path of the test annotation")
    parse.add_argument('--model_name', type=str, default = "CenterNet_ResNet101",help="the name of trained model")
    parse.add_argument("--weights", type=str, default = r'/root/autodl-tmp/Tracking-3D-particles/pretrained_train/resnet101/best_epoch_weights.pth',help="the path to weigths")
    parse.add_argument("--input_size", type=int, default=[1024, 1024], help="the initial image size of the model")
    parse.add_argument("--cuda", default=True,help="whether to use cuda")
    parse.add_argument("--num_classes", type=int, default=2, help="the number of label")
    parse.add_argument("--classes", type=str, default=['small', 'big'])
    parse.add_argument("--save_path", type=str, default=r'\result', help="the path to save the save the prediction result")

    ## the config of single image prediction
    parse.add_argument("--nms_threshold", type=float, default=0.3, help="the threshold to NMS algorithm")
    parse.add_argument("--confidence_threshold", type=float, default=0.3, help="the threshold to confidence")
    parse.add_argument("--box_length", type=int, default=20, help="the length of the box's side")
    args = parse.parse_args()
    return args



def cal_loss(pred_results,gt_results,num_classes):
    pred_results = np.array(pred_results,dtype=np.float32)

    pred_results = torch.tensor(pred_results)
    gt_results = torch.tensor(gt_results)
    unique_labels = pred_results[:,0].cpu().unique()

    _, conf_sort_index = torch.sort(pred_results[:,1], descending=True)
    pred_results = pred_results[conf_sort_index]

    loss_loc_all = []
    loss_depth_all = []
    for i in range(num_classes):
        loss_loc_all.append([])
        loss_depth_all.append([])



    loss_class_sum = np.zeros([num_classes])
    num_particle = np.zeros([num_classes])
    loss_loc_nn=[]
    loss_depth_nn = []

    gt_pair = np.zeros([len(pred_results),5],dtype=float)
    pred_pair = np.zeros([len(pred_results),5], dtype=float)

    mm=-1

    for c in unique_labels:
        m = int(c.item())

        detections_class = pred_results[pred_results[:,0] == c]

        if len(detections_class)>len(gt_results):
            for i in range(len(gt_results)):
                if len(detections_class)==0:
                    break
                else:
                    center_x_preds = detections_class[:,2]
                    center_y_preds = detections_class[:, 3]
                    depth_preds = detections_class[:, 4]
                    center_x_gt = (gt_results[i, 0] + gt_results[i, 2]) / 2
                    center_y_gt = (gt_results[i, 1] + gt_results[i, 3]) / 2
                    gt_class = gt_results[i, 5]
                    mm=mm+1
                    loss_x = abs(center_x_preds - center_x_gt)
                    loss_y = abs(center_y_preds - center_y_gt)
                    loss_loc = (loss_y ** 2 + loss_x ** 2) ** 0.5

                    min_indices = torch.argmin(loss_loc)
                    min = torch.min(loss_loc)
                    class_pred = detections_class[min_indices,0]
                    if min.item()<5:
                        if gt_class == class_pred:
                            num_particle[m] = num_particle[m] + 1
                            min = min.numpy()
                            loss_dd = abs(depth_preds[min_indices] - gt_results[i, 4])
                            loss_dd = loss_dd.numpy()
                            loss_loc_all[m].append(min)
                            loss_depth_all[m].append(loss_dd)
                            loss_loc_nn.append(min)
                            loss_depth_nn.append(loss_dd)
                            gt_pair[mm] = [center_x_gt, center_y_gt, gt_results[i, 4],min,loss_dd]
                            pred_pair[mm] = [center_x_preds[min_indices], center_y_preds[min_indices], depth_preds[min_indices],min,loss_dd]
                        else:
                            loss_class_sum[m] = loss_class_sum[m] + 1
                    else:
                        loss_class_sum[m] = loss_class_sum[m] + 1


                    detections_class = np.delete(detections_class, min_indices, axis=0)
        else:
            for i in range(len(detections_class)):
                if len(gt_results) == 0:
                    break
                else:
                    center_x_gts = (gt_results[:, 0] + gt_results[:, 2]) / 2
                    center_y_gts = (gt_results[:, 1] + gt_results[:, 3]) / 2
                    depth_gts = gt_results[:, 4]
                    mm=mm+1
                    center_x_pred = detections_class[i,2]
                    center_y_pred = detections_class[i,3]
                    class_pred = detections_class[i,0]
                    loss_x = abs(center_x_pred-center_x_gts)
                    loss_y = abs(center_y_pred-center_y_gts)
                    loss_loc = (loss_y**2+loss_x**2)**0.5
                    min_indices = torch.argmin(loss_loc)
                    min = torch.min(loss_loc)
                    gt_class = gt_results[min_indices,5]
                    if min.item()<5:
                        if gt_class == class_pred:
                            num_particle[m] = num_particle[m]+1
                            min = min.numpy()
                            loss_dd = abs(depth_gts[min_indices] - detections_class[i, 4])
                            loss_dd = loss_dd.numpy()
                            loss_loc_all[m].append(min)
                            loss_depth_all[m].append(loss_dd)
                            loss_loc_nn.append(min)
                            loss_depth_nn.append(loss_dd)
                            gt_pair[mm] = [center_x_gts[min_indices],center_y_gts[min_indices],depth_gts[min_indices],min,loss_dd]
                            pred_pair[mm] = [center_x_pred,center_y_pred,detections_class[i,4],min,loss_dd]
                        else:
                            loss_class_sum[m] = loss_class_sum[m]+1
                    else:
                        loss_class_sum[m] = loss_class_sum[m]+1
                    gt_results = np.delete(gt_results, min_indices, axis=0)



    return loss_loc_all,loss_depth_all,loss_class_sum,num_particle,loss_loc_nn,loss_depth_nn,gt_pair,pred_pair




def get_gt_data(annotation_line):
    line = annotation_line.split()
    image = Image.open(line[0])
    box = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]])

    return image,box

if __name__ == '__main__':
    current_path = Path.cwd()

    config_dict = get_config(current_path)
    classes = config_dict['classes'].split(',')
    num_classes = len(classes)


    args = parse()
    if args.model_name == "CenterNet_ResNet50":
        model = CenterNet_ResNet(numclass=num_classes, pretrained=False, resnet_flag="resnet50")
    if args.model_name == "CenterNet_ResNetX":
        model = CenterNet_ResNetX(numclass=num_classes, pretrained=False)
    if args.model_name == "CenterNet_DLA34":
        model = CenterNet_DLA34(numclasses=num_classes, pretrained=False)
    if args.model_name == "CenterNet_ResNet101":
        model = CenterNet_ResNet(numclass=num_classes, pretrained=False, resnet_flag="resnet101")

    if args.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device=device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    with open(args.test_annotation_path) as f:
        data_ground_truthes = f.readlines()

    num_test = len(data_ground_truthes)
    loss_loc_all = []
    loss_depth_all = []
    num_particle_all =0
    loss_loc_total = 0
    loss_depth_total = 0
    loss_class_total = 0
    loss_loc_item = np.zeros([num_classes])
    loss_depth_item = np.zeros([num_classes])
    loss_class_item = np.zeros([num_classes])
    num_particle_all_correct = np.zeros([num_classes])
    miss_all = 0
    loss_loc_class_val = []
    loss_depth_class_val = []

    gt_pair_sum = []
    pred_pair_sum = []

    miss_small=0
    miss_big=0


    for data_ground_truth in data_ground_truthes:
        image,box = get_gt_data(data_ground_truth)
        image_result,infos=detect_single_image(image, args.input_size, model, args.nms_threshold, args.confidence_threshold, args.cuda, args.box_length,
                            letterbox_image=True, count=True, num_classes=num_classes, classes=classes,show_flag=False)
        classes_nums = infos[0][-1]
        infos = [info[0:5] for info in infos]

        loss_loc_nn, loss_depth_nn, loss_class_sum, num_particle, loss_loc, loss_depth, gt_pair, pred_pair = cal_loss(
            infos, box, num_classes)



        loss_class_total += sum(loss_class_sum)

        loss_loc_all.append(loss_loc)
        loss_depth_all.append(loss_depth)

        loss_loc_class_val.append(loss_loc_nn)
        loss_depth_class_val.append(loss_depth_nn)

        gt_pair_sum.append(gt_pair)
        pred_pair_sum.append(pred_pair)

        num_particle_all_correct += num_particle

        miss_small += abs(classes_nums[0] - num_particle[0])
        miss_big += abs(classes_nums[1] - num_particle[1])

    loss_loc_all = [j for i in loss_loc_all for j in i]
    loss_depth_all = [j for i in loss_depth_all for j in i]
    loss_loc_total = sum(loss_loc_all)
    loss_depth_total = sum(loss_depth_all)

    loss_loc_small = [x[0] for x in loss_loc_class_val]
    loss_loc_small = [j for i in loss_loc_small for j in i]
    loss_loc_big = [x[1] for x in loss_loc_class_val]
    loss_loc_big = [j for i in loss_loc_big for j in i]

    loss_depth_small = [x[0] for x in loss_depth_class_val]
    loss_depth_small = [j for i in loss_depth_small for j in i]
    loss_depth_big = [x[1] for x in loss_depth_class_val]
    loss_depth_big = [j for i in loss_depth_big for j in i]

    gt_pair_sum = [j for i in gt_pair_sum for j in i]
    pred_pair_sum = [j for i in pred_pair_sum for j in i]


    print("Total target position error", ":", loss_loc_total / sum(num_particle_all_correct))

    print("Total target depth error", ":", loss_depth_total / sum(num_particle_all_correct))

    print("A total of ",sum(num_particle_all_correct),"balls were judged correctly")

    print("There are",num_particle_all ," balls in total.")

    # 误检
    print('The number of misdetected spheres was', loss_class_total)

    # 大球的位置误差
    print("Positional error of the big ball：", np.mean(loss_loc_big))
    # 大球的深度误差：
    print("Depth error of the big ball：", np.mean(loss_depth_big))

    # 小球的位置误差
    print("Positional error of the small ball：", np.mean(loss_loc_small))
    # 小球的深度误差
    print("Depth error of the small ball：", np.mean(loss_depth_small))

    loss_loc_all = np.array(loss_loc_all)
    loss_depth_all = np.array(loss_depth_all)

    loss_depth_big = np.array(loss_depth_big)
    loss_depth_small = np.array(loss_depth_small)

    loss_loc_small = np.array(loss_loc_small)
    loss_loc_big = np.array(loss_loc_big)

    gt_pair_sum = np.array(gt_pair_sum)
    pred_pair_sum = np.array(pred_pair_sum)


   















































