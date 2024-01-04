import colorsys
import os
import time
import numpy as np
import torch
from PIL import Image

from models.model import CenterNet_Resnet,CenterNet_ConvNext,CenterNet_Dla34,CenterNet_ResnetX
from utils1.utils import cvtColor,get_classes,preprocess_input,resize_image

from utils1.utils_bbox import decode_bbox,postprocess,decode_bbox_pos


def detect_one_image(image,net,cuda,num_classes,class_names,confidence=0.1,count=True,crop=False,letterbox_image=False,nms_iou=0.3,nms=True):
    input_shape= (1024, 1024)
    image_shape = np.array(np.shape(image)[0:2])
    image = cvtColor(image)
    image_data, gw, gh = resize_image(image, (input_shape[1], input_shape[0]), letterbox_image)

    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

    with torch.no_grad():
        images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
        if cuda:
            images = images.cuda()
        outputs = net(images)

        outputs = decode_bbox(outputs[0], outputs[1], outputs[2], gw, gh, confidence, cuda)

        results = postprocess(outputs, nms, image_shape, input_shape, letterbox_image, nms_iou)

        if results[0] is None:
            return image

        top_label = np.array(results[0][:, 6], dtype='int32')
        top_conf = results[0][:, 5]
        top_depth = results[0][:,4]
        top_boxes = results[0][:, :4]

    # ---------------------------------------------------------#
    #   计数
    # ---------------------------------------------------------#
    if count:
        classes_nums = np.zeros([num_classes])
        for i in range(num_classes):
            num = np.sum(top_label == i)


            classes_nums[i] = num
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
    detecttion_result =list()
    for i, c in list(enumerate(top_label)):
        predicted_class = class_names[int(c)]
        box = top_boxes[i]
        score = top_conf[i]
        depth = top_depth[i]
        top, left, bottom, right = box
        top = max(0, top.astype('float32'))
        left = max(0, left.astype('float32'))
        bottom = min(image.size[1],bottom.astype('float32'))
        right = min(image.size[0], right.astype('float32'))
        center_y = (top + bottom) / 2
        center_x = (left + right) / 2

        result=[int(c),score,center_x,center_y,depth]
        detecttion_result.append(result)
    return detecttion_result,classes_nums

def cal_loss(pred_results,gt_results,num_classes):
    pred_results = np.array(pred_results,dtype=np.float32)

    pred_results = torch.tensor(pred_results)
    gt_results = torch.tensor(gt_results)
    unique_labels = pred_results[:,0].cpu().unique()

    _, conf_sort_index = torch.sort(pred_results[:,1], descending=True)
    pred_results = pred_results[conf_sort_index]

    loss_loc_all = []
    for i in range(num_classes):
        loss_loc_all.append([])

    loss_depth_all = []
    for i in range(num_classes):
        loss_depth_all.append([])

    loss_class_sum = np.zeros([num_classes])
    num_particle = np.zeros([num_classes])
    loss_loc_nn=[]
    loss_depth_nn = []

    gt_pair = np.zeros([len(pred_results),6],dtype=float)
    pred_pair = np.zeros([len(pred_results),6], dtype=float)

    mm=-1

    for m, c in enumerate(unique_labels):

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
                    if min.item()<10:
                        if gt_class == class_pred:
                            num_particle[m] = num_particle[m] + 1
                            min = min.numpy()
                            loss_dd = abs(depth_preds[min_indices] - gt_results[i, 4])
                            loss_dd = loss_dd.numpy()
                            loss_loc_all[m].append(min)
                            loss_depth_all[m].append(loss_dd)
                            loss_loc_nn.append(min)
                            loss_depth_nn.append(loss_dd)
                            gt_pair[mm] = [center_x_gt, center_y_gt, gt_results[i, 4],min,loss_dd,gt_class]
                            pred_pair[mm] = [center_x_preds[min_indices], center_y_preds[min_indices], depth_preds[min_indices],min,loss_dd,gt_class]
                        else:
                            loss_class_sum[m] = loss_class_sum[m] + 1
                    else:
                        break
                    detections_class = np.delete(detections_class, min_indices, axis=0)
        else:
            for i in range(len(detections_class)):
                if len(gt_results) == 0:
                    break
                else:
                    center_x_gts = (gt_results[:, 0] + gt_results[:, 2]) / 2.000
                    center_y_gts = (gt_results[:, 1] + gt_results[:, 3]) / 2.000
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
                    if min.item()<10:
                        if gt_class == class_pred:
                            num_particle[m] = num_particle[m]+1
                            min = min.numpy()
                            loss_dd = abs(depth_gts[min_indices] - detections_class[i, 4])
                            loss_dd = loss_dd.numpy()
                            loss_loc_all[m].append(min)
                            loss_depth_all[m].append(loss_dd)
                            loss_loc_nn.append(min)
                            loss_depth_nn.append(loss_dd)
                            gt_pair[mm] = [center_x_gts[min_indices],center_y_gts[min_indices],depth_gts[min_indices],min,loss_dd,class_pred]
                            pred_pair[mm] = [center_x_pred,center_y_pred,detections_class[i,4],min,loss_dd,class_pred]
                        else:
                            loss_class_sum[m] = loss_class_sum[m]+1
                    else:
                        break


                    gt_results = np.delete(gt_results, min_indices, axis=0)

    return loss_loc_all,loss_depth_all,loss_class_sum,num_particle,loss_loc_nn,loss_depth_nn,gt_pair,pred_pair

def generate_net(backbone,num_classes,pretrained=False):
    if backbone == "resnet101":
        model = CenterNet_Resnet(num_classes, pretrained=pretrained)
    if backbone =="resnet50X":
        model = CenterNet_ResnetX(num_classes,pretrained=pretrained)
    if backbone == 'convnext':
        model = CenterNet_ConvNext(num_classes, pretrained=pretrained)
    if backbone == 'dla':
        model = CenterNet_Dla34(num_classes)
        return model

def get_gt_data(annotation_line):
    line = annotation_line.split()
    image = Image.open(line[0])
    box = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]])

    return image,box



if __name__=="__main__":
    test_annotation_path = 'database/test_real.txt'
    classes_path = 'database/classes.txt'
    model_path  = "logs\\resnet50x\\last_epoch_weights.pth"
    Cuda = True
    if Cuda:

        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    class_names, num_classes = get_classes(classes_path)

    net = generate_net('resnet50X',num_classes)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net = net.eval()
    if Cuda:
        net = torch.nn.DataParallel(net)
        net = net.cuda()

    with open(test_annotation_path) as f:
        test_lines = f.readlines()

    num_test = len(test_lines)
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

    for i in range(num_test):

          image,box = get_gt_data(test_lines[i])
          num_particle_all = num_particle_all+len(box)
          detecttion_result,classes_nums=detect_one_image(image, net, Cuda, num_classes, class_names, confidence=0.1, count=True, crop=False,
                           letterbox_image=False, nms_iou=0.3, nms=True)
          loss_loc_nn,loss_depth_nn,loss_class_sum,num_particle,loss_loc,loss_depth,gt_pair,pred_pair=cal_loss(detecttion_result, box, num_classes)

          loss_class_total += sum(loss_class_sum)

          loss_loc_all.append(loss_loc)
          loss_depth_all.append(loss_depth)

          loss_loc_class_val.append(loss_loc_nn)
          loss_depth_class_val.append(loss_depth_nn)

          gt_pair_sum.append(gt_pair)
          pred_pair_sum.append(pred_pair)


          num_particle_all_correct += num_particle

          miss_small += abs(classes_nums[0]-num_particle[0])
          miss_big += abs(classes_nums[1]-num_particle[1])

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



    print("总的目标位置误差",":",loss_loc_total/sum(num_particle_all_correct))

    print("总的目标深度误差",":",loss_depth_total/sum(num_particle_all_correct))


    print("总共正确判断了",sum(num_particle_all_correct),"个目标")

    print("总共有", num_particle_all, "个目标")


    print('误检目标数为',loss_class_total)


    print("大球的位置误差：",np.mean(loss_loc_big))

    print("大球的深度误差：", np.mean(loss_depth_big))


    print("小球的位置误差：",np.mean(loss_loc_small))

    print("小球的深度误差：",np.mean(loss_depth_small))

    loss_loc_all = np.array(loss_loc_all)
    loss_depth_all = np.array(loss_depth_all)

    loss_depth_big = np.array(loss_depth_big)
    loss_depth_small = np.array(loss_depth_small)

    loss_loc_small = np.array(loss_loc_small)
    loss_loc_big = np.array(loss_loc_big)

    gt_pair_sum = np.array(gt_pair_sum)
    pred_pair_sum = np.array(pred_pair_sum)

    np.savetxt('', loss_loc_all, delimiter=",")

    np.savetxt('', loss_depth_all, delimiter=",")

    np.savetxt('', loss_depth_small, delimiter=",")
    np.savetxt('', loss_depth_big, delimiter=",")

    np.savetxt('', loss_loc_small, delimiter=",")
    np.savetxt('', loss_loc_big, delimiter=",")

    np.savetxt('', gt_pair_sum, delimiter=",")
    np.savetxt('', pred_pair_sum, delimiter=",")

















