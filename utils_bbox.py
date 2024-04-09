import numpy as np
from torch import nn
import torch

def pool_nms(pred_hms,kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(pred_hms, kernel_size=(kernel,kernel),stride=1,padding=pad)
    keep = (hmax==pred_hms).float()
    return keep*pred_hms

def decode_bbox(pred_hms,pred_depths,pred_offsets,gw,gh,confidence_threshold,cuda):
    pred_hms = pool_nms(pred_hms)
    b,c,output_h,output_w = pred_hms.shape
    detects = []

    for batch in range(b):
        heatmap = pred_hms[batch].permute(1,2,0).view([-1,c])
        depths = pred_depths[batch].permute(1,2,0).view([-1,1])
        offsets = pred_offsets[batch].permute(1,2,0).view([-1,c])

        yv,xv = torch.meshgrid(torch.arange(0,output_h),torch.arange(0,output_w))
        xv,yv = xv.flatten().float(),yv.flatten().float()
        if cuda:
            xv = xv.cuda()
            yv = yv.cuda()

        class_conf,class_pred = torch.max(heatmap,dim=-1)
        mask = class_conf > confidence_threshold

        pred_depths_mask = depths[mask]
        pred_offsets_mask = offsets[mask]

        if len(pred_depths_mask) == 0:
            detects.append([])
            continue

        # centers' coordinate
        xv_mask = torch.unsqueeze(xv[mask]+pred_offsets_mask[...,0],-1)
        yv_mask = torch.unsqueeze(yv[mask]+pred_offsets_mask[...,1],-1)

        pred_depths_mask = pred_depths_mask*50
        # boxes' coordinate
        bboxes = torch.cat([xv_mask-float(gw/2),yv_mask-float(gh/2),xv_mask+float(gw/2),yv_mask+float(gh/2)],dim=1)
        bboxes[:,[0,2]] = bboxes[:,[0,2]]/output_w
        bboxes[:,[1,3]] = bboxes[:,[1,3]]/output_h

        detect = torch.cat([bboxes,pred_depths_mask,torch.unsqueeze(class_conf[mask],-1),torch.unsqueeze(class_pred[mask],-1)],dim=-1)
        detects.append(detect)

    return detects

def bbox_iou(bboxes1,bboxes2):
    b1_x1,b1_y1,b1_x2,b1_y2 = bboxes1[:,0],bboxes1[:,1],bboxes1[:,2],bboxes1[:,3]
    b2_x1,b2_y1,b2_x2,b2_y2 = bboxes2[:,0],bboxes2[:,1],bboxes2[:,2],bboxes2[:,3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / torch.clamp(b1_area + b2_area - inter_area, min=1e-6)

    return iou

def centernet_correct_boxes(box_xy, box_wh, input_size, image_shape,letterbox_image):
    box_yx = box_xy[...,::-1]
    box_hw = box_wh[...,::-1]
    input_shape = np.array(input_size)
    image_shape = np.array(image_shape)
    if letterbox_image:
        new_shape = np.round(image_shape * np.min(input_shape/image_shape))
        offset = (input_shape-new_shape)/2./input_shape[0]
        scale = input_shape/new_shape
        box_yx = (box_yx-offset)*scale
        box_hw = box_wh*scale

    box_mins = box_yx - (box_hw / 2)
    box_maxs = box_yx + (box_hw / 2)
    boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxs[..., 0:1], box_maxs[..., 1:2]],axis=-1)
    boxes *= np.concatenate([image_shape, image_shape],axis=-1)
    return boxes





def postprocess(predictions,image_shape,input_size,nms_threshold):
    outputs = [None for _ in range(len(predictions))]
    for i,prediction in enumerate(predictions):
        if len(prediction)==0:
            continue
        unique_labels = prediction[...,-1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
            prediction = prediction.cuda()

        for c in unique_labels:
            detection_class_info = prediction[prediction[:,-1]==c]
            _,conf_sort_index = torch.sort(detection_class_info[:,5],descending=True)
            detection_class_info = detection_class_info[conf_sort_index]

            max_detection = []

            while detection_class_info.size(0)>0:
                max_detection.append(detection_class_info[0].unsqueeze(0))
                if len(detection_class_info) == 1:
                    break
                ious = bbox_iou(max_detection[-1],detection_class_info[1:])
                detection_class_info = detection_class_info[1:][ious<nms_threshold]

            max_detection = torch.cat(max_detection).data

            outputs[i] = max_detection if outputs[i] is None else torch.cat((outputs[i],max_detection))

        if outputs[i] is not None:
            outputs[i] = outputs[i].cpu().numpy()
            box_xy,box_wh = (outputs[i][:, 0:2] + outputs[i][:, 2:4])/2, outputs[i][:, 2:4] - outputs[i][:, 0:2]
            outputs[i][:,:4] = centernet_correct_boxes(box_xy, box_wh, input_size, image_shape,letterbox_image=True)

    return outputs






































