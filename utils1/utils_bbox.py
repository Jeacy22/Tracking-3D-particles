import numpy as np
import torch
from torch import nn
import cv2


def pool_nms(heat, kernel = 3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep
def get_max_preds(batch_heatmaps,confidence):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''

    batch_heatmaps=torch.from_numpy(batch_heatmaps)
    num = batch_heatmaps.shape[0]
    heatmaps_reshaped = batch_heatmaps.reshape((-1))
    vals,_ = torch.sort(heatmaps_reshaped)
    vals = vals.cpu().numpy()

    mask = vals>confidence
    vals = vals[mask]

    return  vals

def gaussian_blur(hm, kernel,):
    border = (kernel - 1) // 2
    num = hm.shape[0]
    height = hm.shape[1]
    width = hm.shape[2]
    hm=hm.cpu().numpy()

    for j in range(num):
        origin_max = np.max(hm[j,:,:])
        dr = np.zeros((height + 2 * border, width + 2 * border))
        dr[border: -border, border: -border] = hm[j,:,:].copy()
        dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
        hm[j,:,:] = dr[border: -border, border: -border].copy()
        hm[j,:,:] *= origin_max / np.max(hm[j])
    return hm

def taylor(hm, coord, pred_offset,pred_depth,class_conf,class_pred):
    heatmap_height = hm.shape[0]
    heatmap_width = hm.shape[1]
    coord_result = np.zeros((2), dtype=float)
    px = int(coord[1])
    py = int(coord[0])
    if 1 < px < heatmap_width-2 and 1 < py < heatmap_height-2:
        dx  = 0.5 * (hm[py][px+1] - hm[py][px-1])
        dy  = 0.5 * (hm[py+1][px] - hm[py-1][px])
        dxx = 0.25 * (hm[py][px+2] - 2 * hm[py][px] + hm[py][px-2])
        dxy = 0.25 * (hm[py+1][px+1] - hm[py-1][px+1] - hm[py+1][px-1] \
            + hm[py-1][px-1])
        dyy = 0.25 * (hm[py+2*1][px] - 2 * hm[py][px] + hm[py-2*1][px])
        derivative = np.matrix([[dx],[dy]])
        hessian = np.matrix([[dxx,dxy],[dxy,dyy]])
        if dxx * dyy - dxy ** 2 != 0:
            hessianinv = hessian.I
            offset = -hessianinv * derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord_result[0] = offset[0]+px+pred_offset[0,py,px].cpu().numpy()
            coord_result[1] = offset[0]+py+pred_offset[1,py,px].cpu().numpy()
            depth = pred_depth[0,py,px].cpu().numpy()
            conf = class_conf[py,px]
            class1 = class_pred[py, px]
    return coord_result,depth,conf,class1

def decode_bbox_pos(pred_hms, pred_depths, pred_offsets, gw,gh,confidence, cuda):

    pred_hms = pool_nms(pred_hms)


    for batch in range(pred_hms.size(0)):
        pred_offset = pred_offsets[batch]
        pred_depth = pred_depths[batch]
        #hm = pred_hms[batch]
        #hm = hm.cpu().numpy()
        hm = gaussian_blur(pred_hms[batch], 11)

        hm = np.maximum(hm, 1e-10)
        different = hm[1] - hm[0]
        class_pred = np.int64(different > 0)
        class_conf = np.maximum(hm[0], hm[1])
        maxvals = get_max_preds(class_conf,confidence)

        coor = np.zeros((maxvals.shape[0], 2), dtype=float)
        depth = np.zeros((maxvals.shape[0], 1), dtype=float)
        conf = np.zeros((maxvals.shape[0], 1), dtype=float)
        class1 = np.zeros((maxvals.shape[0], 1), dtype=float)




        for m in range(maxvals.shape[0]):
            pos =np.where(class_conf==maxvals[m])
            pos = np.array(list(pos)).astype(dtype=int).tolist()
            pos = np.hstack(pos).tolist()
            if len(pos)==2:
                coor[m,:],depth[m,:],conf[m,:],class1[m,:]= taylor(class_conf,pos,pred_offset,pred_depth,class_conf,class_pred)
            else:
                mid = int(len(pos)/2)
                pos_n = [pos[1],pos[mid+1]]
                coor[m, :], depth[m, :], conf[m, :], class1[m, :] = taylor(class_conf, pos_n, pred_offset, pred_depth,
                                                                           class_conf, class_pred)




    b, c, output_h, output_w = pred_hms.shape
    detects = []

    for batch in range(b):

        xv_mask = torch.from_numpy(coor[:,0]).unsqueeze(1)
        yv_mask = torch.from_numpy(coor[:,1]).unsqueeze(1)
        depth = torch.from_numpy(depth)
        conf = torch.from_numpy(conf)
        class1 = torch.from_numpy(class1)
        depth = depth[..., 0:1]*50

        bboxes = torch.cat([xv_mask - float(gw/2), yv_mask - float(gh/2), xv_mask + float(gw/2), yv_mask +float(gh/2)], dim=1)
        bboxes[:, [0, 2]] /= output_w
        bboxes[:, [1, 3]] /= output_h
        detect = torch.cat([bboxes, depth,conf, class1], dim=-1)
        detects.append(detect)

    return detects

def decode_bbox(pred_hms, pred_depths, pred_offsets, gw,gh,confidence, cuda):

    pred_hms = pool_nms(pred_hms)
    
    b, c, output_h, output_w = pred_hms.shape
    detects = []

    for batch in range(b):

        heat_map    = pred_hms[batch].permute(1, 2, 0).view([-1, c])
        pred_depth     = pred_depths[batch].permute(1, 2, 0).view([-1, 1])
        pred_offset = pred_offsets[batch].permute(1, 2, 0).view([-1, 2])

        yv, xv      = torch.meshgrid(torch.arange(0, output_h), torch.arange(0, output_w))

        xv, yv      = xv.flatten().float(), yv.flatten().float()
        if cuda:
            xv      = xv.cuda()
            yv      = yv.cuda()


        class_conf, class_pred  = torch.max(heat_map, dim = -1)
        mask                    = class_conf > confidence


        pred_depth_mask        = pred_depth[mask]
        pred_offset_mask    = pred_offset[mask]
        if len(pred_offset_mask) == 0:
            detects.append([])
            continue     


        xv_mask = torch.unsqueeze(xv[mask] + pred_offset_mask[..., 0], -1)
        yv_mask = torch.unsqueeze(yv[mask] + pred_offset_mask[..., 1], -1)

        depth = pred_depth_mask[..., 0:1]*50

        bboxes = torch.cat([xv_mask - float(gw/2), yv_mask - float(gh/2), xv_mask + float(gw/2), yv_mask +float(gh/2)], dim=1)
        bboxes[:, [0, 2]] /= output_w
        bboxes[:, [1, 3]] /= output_h
        detect = torch.cat([bboxes, depth,torch.unsqueeze(class_conf[mask],-1), torch.unsqueeze(class_pred[mask],-1).float()], dim=-1)
        detects.append(detect)

    return detects

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
        计算IOU
    """
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)*10
    inter_rect_y1 = torch.max(b1_y1, b2_y1)*10
    inter_rect_x2 = torch.min(b1_x2, b2_x2)*10
    inter_rect_y2 = torch.min(b1_y2, b2_y2)*10

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    iou = inter_area / torch.clamp(b1_area + b2_area - inter_area, min = 1e-6)

    return inter_area

def calculate_dis(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    centerx1 = (b1_x1+b1_x2)/2
    centery1 = (b1_y1 + b1_y2) / 2

    centerx2 = (b2_x1 + b2_x2) / 2
    centery2 = (b2_y1 + b2_y2) / 2

    scale1 = abs(centerx2-centerx1)/abs(b2_x2-centerx1)
    scale2 = abs(centery2 - centery1) / abs(b2_y2 - centery1)


    return scale1

def centernet_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)

    if letterbox_image:

        new_shape = np.round(image_shape * np.min(input_shape/image_shape))
        offset  = (input_shape - new_shape)/2./input_shape
        scale   = input_shape/new_shape

        box_yx  = (box_yx - offset) * scale
        box_hw *= scale

    box_mins    = box_yx - (box_hw / 2.)
    box_maxes   = box_yx + (box_hw / 2.)
    boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes

def postprocess(prediction, need_nms, image_shape, input_shape, letterbox_image, nms_thres=0):
    output = [None for _ in range(len(prediction))]
    

    for i, image_pred in enumerate(prediction):
        detections      = prediction[i]
        if len(detections) == 0:
            continue

        unique_labels   = detections[:, -1].cpu().unique()

        #pos = np.where(detections[:, 5].cpu() < 0.2)
        #pos = np.array(list(pos)).astype(dtype=int).tolist()
        #detections = np.delete(detections.cpu(), pos[0], axis=0)
        if detections.is_cuda:
            unique_labels = unique_labels.cuda()
            detections = detections.cuda()


        for c in unique_labels:

            detections_class = detections[detections[:, -1] == c]
            if need_nms:

                """
                 keep = nms(
                    detections_class[:, :4],
                    detections_class[:, 4],
                    nms_thres
                )
                max_detections = detections_class[keep]
                """

                _, conf_sort_index = torch.sort(detections_class[:, 5], descending=True)
                detections_class = detections_class[conf_sort_index]

                max_detections = []
                while detections_class.size(0):

                     max_detections.append(detections_class[0].unsqueeze(0))
                     if len(detections_class) == 1:
                         break
                     ious = bbox_iou(max_detections[-1], detections_class[1:])

                     detections_class = detections_class[1:][ious == nms_thres]

                     #dis = calculate_dis(max_detections[-1], detections_class[1:])
                     #detections_class = detections_class[1:][dis>1]

                max_detections = torch.cat(max_detections).data

            else:
                max_detections  = detections_class
            
            output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

        if output[i] is not None:
            output[i]           = output[i].cpu().numpy()
            box_xy, box_wh      = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
            output[i][:, :4]    = centernet_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
    return output
