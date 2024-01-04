
import torch
from utils1.utils import download_weights,get_classes,show_config
from models.model import CenterNet_Resnet,CenterNet_ConvNext,CenterNet_Dla34,CenterNet_ResnetX
import numpy as np

import datetime

import torch.backends.cudnn as cudnn
import torch.optim as optim
from database.dataloader import CenternetDataset, centernet_dataset_collate
from torch.utils.data import DataLoader
from utils1.utils_fit import fit_one_epoch
from models.centernet_training import get_lr_scheduler, set_optimizer_lr
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    Cuda = True
    fp16 = True
    classes_path = 'database/classes.txt'
    model_path = ''
    input_shape = [1024, 1024]
    backbone = "dlax"
    pretrained = True
    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 16
    UnFreeze_Epoch = 400
    Unfreeze_batch_size = 8
    Freeze_Train = True
    Init_lr = 5e-4
    Min_lr = Init_lr * 0.01
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 1e-8
    lr_decay_type = 'cos'
    save_period = 1
    save_dir = 'logs'
    eval_flag = True
    eval_period = 1
    num_workers =4
    train_annotation_path = 'database/train.txt'
    val_annotation_path = 'database/val.txt'

    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    ngpus_per_node = torch.cuda.device_count()
    if Cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")




    class_names,num_classes = get_classes(classes_path)

    if backbone == "resnet101":
        model = CenterNet_Resnet(num_classes, pretrained=pretrained)
    if backbone == "resnet50X":
        model = CenterNet_ResnetX(num_classes, pretrained=pretrained)
    if backbone == 'dla':
        model = CenterNet_Dla34(num_classes)
    if backbone == 'convnext':
        model = CenterNet_ConvNext(num_classes, pretrained=pretrained)





    if model_path != '':
        print('Load weights {}.'.format(model_path))
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        print("\n\033[1;33;44m[Warning],It is normal that the Head part is not loaded, "
              "and it is an error that the Backbone part is not loaded.\033[0m")

    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    logboard_dir = os.path.join(save_dir,"loss_board_"+str(time_str))
    writer = SummaryWriter(logboard_dir)

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()

    num_train = len(train_lines)
    num_val = len(val_lines)

    show_config(
        classes_path=classes_path, model_path=model_path, input_shape=input_shape, \
        Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
        Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train, \
        Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum, lr_decay_type=lr_decay_type, \
        save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
    )

    wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e2
    total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
    if total_step <= wanted_step:
        if num_train // Unfreeze_batch_size == 0:
            raise ValueError('The dataset is too small for training, please expand the dataset.')
        wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
        print("\n\033[1;33;44m[Warning] When using the %s , "
              "it is recommended to set the total training step size above %d .\033[0m" % (
        optimizer_type, wanted_step))
        print(
            "\033[1;33;44m[Warning] The total training data volume for this run is %d, "
            "Unfreeze_batch_size is %d, a total of %d Epochs are trained, "
            "and the total training step size is calculated as %d.\033[0m" % (
            num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
        print("\033[1;33;44m[Warning] Since the total training step is %d, which is less than the recommended total step %d, "
              "it is recommended to set the total Epoch to %d.\033[0m" % (
        total_step, wanted_step, wanted_epoch))


    if True:
        UnFreeze_flag = False
        if backbone =='dla' :
            Freeze_Train = False
            UnFreeze_flag = True
            Freeze_Epoch = 0
            UnFreeze_Epoch = 400

        if Freeze_Train:
            model.freeze_backbone()

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs = 64
        lr_limit_max = 5e-4 if optimizer_type == 'adam' else 5e-2
        lr_limit_min = 2.5e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                             weight_decay=weight_decay)
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        train_sampler = None
        val_sampler = None
        shuffle = True

        epoch_step = num_train//batch_size
        epoch_step_val = num_val//batch_size

        if epoch_step==0 or epoch_step_val ==0:
            raise ValueError("Data set is too small")

        train_dataset = CenternetDataset(train_lines,input_shape,num_classes,train=True)
        val_dataset = CenternetDataset(val_lines,input_shape,num_classes,train=False)

        gen = DataLoader(train_dataset, shuffle=None, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True,drop_last=True, collate_fn=centernet_dataset_collate, sampler=None)
        gen_val = DataLoader(val_dataset, shuffle=None, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,drop_last=True, collate_fn=centernet_dataset_collate, sampler=None)




        for epoch in range(Init_Epoch,UnFreeze_Epoch):

            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                nbs = 64
                lr_limit_max = 5e-4 if optimizer_type == 'adam' else 5e-2
                lr_limit_min = 2.5e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                optimizer = {
                    'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999),
                                       weight_decay=weight_decay),
                    'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                                     weight_decay=weight_decay)
                }[optimizer_type]
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                gen = DataLoader(train_dataset, shuffle=None, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True, drop_last=True, collate_fn=centernet_dataset_collate, sampler=None)
                gen_val = DataLoader(val_dataset, shuffle=None, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True, drop_last=True, collate_fn=centernet_dataset_collate,
                                     sampler=None)
                model.unfreeze_backbone()

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("The dataset is too small to continue the training, please expand the dataset.")

                UnFreeze_flag = True

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, writer, optimizer, epoch,
                              epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler,
                              save_dir, 0)









































