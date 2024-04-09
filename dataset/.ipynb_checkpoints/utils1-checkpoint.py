import os
import logging
import datetime
import os.path
import numpy as np
import torch
import torch.nn.functional as F
import math
from functools import partial


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.05, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0 + math.cos(
                math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def get_config(path):
    config_dict = dict()
    config_path = os.path.join(path, "config.txt")
    with open(config_path) as f:
        data_infos = f.readlines()
    for info in data_infos:
        if len(info) < 2 or info.startswith("#"):
            continue
        key = info.split("=")[0]
        value = info.split("=")[1].replace("\n", "")
        config_dict[key] = value
    return config_dict


def get_logger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    screenhandler = logging.StreamHandler()
    time_now = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    filepath = os.path.join(path, "logger" + str(time_now) + ".txt")
    filehandler = logging.FileHandler(filepath)

    formattern = logging.Formatter("%(message)s")

    screenhandler.setFormatter(formattern)
    filehandler.setFormatter(formattern)

    logger.addHandler(screenhandler)
    logger.addHandler(filehandler)

    return logger


def focal_loss(pred, target):
    pred = pred.permute(0, 2, 3, 1)

    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()

    neg_weights = torch.pow(1 - target, 4)

    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos

    return loss


def reg_l1_loss(pred, target, mask):
    pred = pred.permute(0, 2, 3, 1)
    expand_mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 2)

    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
    loss = loss / (mask.sum())

    return loss


def trainval_one_epoch(train_datasets, val_datasets,
                       model, apm, optimizer, writer, epoch, iter_num_train,
                       iter_num_val, scaler, cuda,  best_val_loss,
                       weigth_path):
    train_hms_loss = 0
    train_offsets_loss = 0
    train_depths_loss = 0
    train_pos_loss = 0
    train_loss = 0

    for i, dataset in enumerate(train_datasets):
        model.train()
        images, batch_hms, batch_depths, batch_offsets, batch_masks = dataset

        with torch.no_grad():
            if cuda:
                images = images.cuda()
                batch_hms = batch_hms.cuda()
                batch_masks = batch_masks.cuda()
                batch_depths = batch_depths.cuda()
                batch_offsets = batch_offsets.cuda()

        optimizer.zero_grad()

        if not apm:
            hms, depths, offsets = model(images)
            hms_loss = focal_loss(hms, batch_hms)
            depths_loss = reg_l1_loss(depths, batch_depths, batch_masks)
            offsets_loss = reg_l1_loss(offsets, batch_offsets, batch_masks)
            loss = hms_loss + depths_loss + offsets_loss
            loss.backward()
            optimizer.step()



        else:
            from torch.cuda.amp import autocast
            with autocast():
                hms, depths, offsets = model(images)
                hms_loss = focal_loss(hms, batch_hms)
                depths_loss = reg_l1_loss(depths, batch_depths, batch_masks)
                offsets_loss = reg_l1_loss(offsets, batch_offsets, batch_masks)
                loss = hms_loss + depths_loss + offsets_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        train_hms_loss = train_hms_loss + hms_loss.item()
        train_offsets_loss = train_offsets_loss + offsets_loss.item()
        train_depths_loss = train_depths_loss + depths_loss.item()
        train_pos_loss = train_pos_loss + offsets_loss.item() + hms_loss.item()
        train_loss = train_loss + loss.item()

    writer.add_scalar("train_loss", train_loss / iter_num_train, epoch + 1)
    writer.add_scalar("train_hms_loss", train_hms_loss / iter_num_train, epoch + 1)
    writer.add_scalar("train_offsets_loss", train_offsets_loss / iter_num_train, epoch + 1)
    writer.add_scalar("train_pos_loss", train_pos_loss / iter_num_train, epoch + 1)
    writer.add_scalar("train_depths_loss", train_depths_loss / iter_num_train, epoch + 1)
    writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch + 1)
    print(
        'train_loss: %.3f || train_hms_loss: %.3f || train_offsets_loss %.3f || train_pos_loss %.3f || '
        'train_depths_loss %.3f' % (
            train_loss / iter_num_train, train_hms_loss / iter_num_train, train_offsets_loss / iter_num_train,
            train_pos_loss / iter_num_train, train_depths_loss / iter_num_train))

    model.eval()
    val_hms_loss = 0
    val_offsets_loss = 0
    val_depths_loss = 0
    val_pos_loss = 0
    val_loss = 0

    for dataset in val_datasets:
        images, batch_hms, batch_depths, batch_offsets, batch_masks = dataset

        with torch.no_grad():

            if cuda:
                images = images.cuda()
                batch_hms = batch_hms.cuda()
                batch_masks = batch_masks.cuda()
                batch_depths = batch_depths.cuda()
                batch_offsets = batch_offsets.cuda()

            hms, depths, offsets = model(images)
            hms_loss = focal_loss(hms, batch_hms)
            depths_loss = reg_l1_loss(depths, batch_depths, batch_masks)
            offsets_loss = reg_l1_loss(offsets, batch_offsets, batch_masks)

            val_hms_loss = val_hms_loss + hms_loss.item()
            val_depths_loss = val_depths_loss + depths_loss.item()
            val_offsets_loss = val_offsets_loss + offsets_loss.item()
            val_pos_loss = val_pos_loss + hms_loss.item() + offsets_loss.item()
            val_loss = val_loss + hms_loss.item() + depths_loss.item() + offsets_loss.item()

    if val_loss < best_val_loss:
        torch.save(model.state_dict(), os.path.join(weigth_path, "weigth.pth"))

    writer.add_scalar("val_pos_loss", val_pos_loss / iter_num_val, epoch + 1)
    writer.add_scalar("val_hms_loss", val_hms_loss / iter_num_val, epoch + 1)
    writer.add_scalar("val_offsets_loss", val_offsets_loss / iter_num_val, epoch + 1)
    writer.add_scalar("val_depths_loss", val_depths_loss / iter_num_val, epoch + 1)
    writer.add_scalar("val_loss", val_loss / iter_num_val, epoch + 1)

    print(
        'val_loss: %.3f || val_hms_loss: %.3f || val_offsets_loss %.3f || val_pos_loss %.3f || val_offsets_loss %.3f' % (
            val_loss / iter_num_val, val_hms_loss / iter_num_val, val_offsets_loss / iter_num_val,
            val_pos_loss / iter_num_val, val_depths_loss / iter_num_val))
