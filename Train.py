import argparse
import os.path
from torch.utils.data import DataLoader
from utils import get_logger,get_config,trainval_one_epoch,set_optimizer_lr,get_lr_scheduler
import torch
import numpy as np
import random
from model.models import CenterNet_ResNet,CenterNet_ResNetX,CenterNet_DLA34
from dataset.dataloader import MyDataSet,centernet_dataset_collate
import datetime
from torch.utils.tensorboard import SummaryWriter
from thop import profile




def lr_set(init_lr,min_lr,batch_size,optimizer_type,model,weight_decay,epoch):
    nbs = 8
    lr_limit_max = 5e-4 if optimizer_type == 'adam' else 5e-2
    lr_limit_min = 2.5e-4 if optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    optimizer = {
        'adam': torch.optim.Adam(model.parameters(), Init_lr_fit, betas=(0.9, 0.999), weight_decay=weight_decay),
        'sgd': torch.optim.SGD(model.parameters(), Init_lr_fit, momentum=0.9, nesterov=True,
                               weight_decay=weight_decay)
    }[optimizer_type]

    scheduler_lr=get_lr_scheduler(lr_decay_type='cos',lr=Init_lr_fit,min_lr=Min_lr_fit,total_iters=epoch)

    return optimizer,scheduler_lr


def parse_args():
    parser=argparse.ArgumentParser(description="Train tracking network")
    parser.add_argument("--cuda",default=True,help="whether using GPU")
    parser.add_argument("--amp",default=True,help="whether using AMP")
    parser.add_argument("--seed",default=43,help="random seed")

    parser.add_argument("--weight_path",default=r"",help="the path to weight_file")
    parser.add_argument("--weight_save_path",default=r"/root/autodl-tmp/Tracking-3D-particles/weight_saved/",help="the path to save weight parameters")

    parser.add_argument("--init_lr",default=5e-5,help="the initial learning rate")
    parser.add_argument("--min_lr",default=5e-6,help="the minimal learning rate")
    parser.add_argument("--dataset_type",default="user_defined",help="what kind of dataset do you want to use")
    parser.add_argument("--optimizer_type",default="adam",help="what kind of optimizer do you want to use")
    parser.add_argument("--pretrained",default=True,help="whether to use pretrained model")
    parser.add_argument("--freezing_epoch",default=50,help="the amount of freezeing epoch")
    parser.add_argument("--freezing_flag",default=True,help="whether to freeze the network")
    parser.add_argument("--freezing_batch_size", default=16)
    parser.add_argument("--unfreezing_batch_size",default=8)
    parser.add_argument("--model_type",default="resnet50",help="choose a kind of model to train")
    parser.add_argument("--image_shape", default=[1024,1024], help="the shape of the image")
    parser.add_argument("--class_amount", default=2, help="how many classes do you want to anaylse")
    parser.add_argument("--num_workes", default=8)
    parser.add_argument("--weight_decay", default=1e-7)
    parser.add_argument("--epoch", default=400)
    args = parser.parse_args()
    return args



def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def train():
    config = get_config(os.getcwd())
    args = parse_args()
    if args.cuda==True and torch.cuda.is_available():
        device = torch.device("cuda")
        set_seed(args.seed)
        if args.amp:
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
        else:
            scaler = None
    else:
        device = torch.device("cpu")



    if args.model_type == "resnet50":
        model = CenterNet_ResNet(numclass=args.class_amount,pretrained=True,resnet_flag="resnet50")
    elif args.model_type=="resnet101":
        model = CenterNet_ResNet(numclass=args.class_amount,pretrained=True,resnet_flag="resnet101")
    elif args.model_type=="resnetx":
        model = CenterNet_ResNetX(numclass=args.class_amount,pretrained=True)
    elif args.model_type=="dla34":
        model = CenterNet_DLA34(numclasses=args.class_amount,pretrained=False)





    model = model.to(device=device)

    if args.dataset_type == "user_defined":
        with open(config["train_dataset"]) as f:
            train_annotation = f.readlines()
            if train_annotation[-1] == "":
                del train_annotation[-1]
        num_trainset = len(train_annotation)


        with open(config["val_dataset"]) as f:
            val_annotation = f.readlines()
            if val_annotation[-1] == "":
                del val_annotation[-1]
        num_testset = len(val_annotation)

    train_dataset=MyDataSet(data_annotation=train_annotation,shape=args.image_shape,
                            classes_amount=args.class_amount,flag=True)
    val_dataset = MyDataSet(data_annotation=val_annotation,shape=args.image_shape,
                            classes_amount=args.class_amount,flag=False)


    if args.pretrained and os.path.exists(args.weight_path):
        static_dict = model.state_dict()
        new_static_dict = torch.load(args.weight_path,map_location=device)
        for key,value in new_static_dict.items():
            if key in static_dict.keys():
                if static_dict[key].shape == new_static_dict[key].shape:
                    static_dict[key] = new_static_dict[key]

        model.load_state_dict(static_dict)
        print("pretrained weights has been loaded")


    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    os.mkdir(time_str)
    print("Log folder has been created")
    logger = get_logger(time_str)
    logger.debug(f"The details of training：\nepoch:{args.epoch}\npretrained:{args.pretrained}\n"
                 f"freezing_flag:{args.freezing_flag}\nfreezing_epoch:{args.freezing_epoch}\n"
                 f"unfreezing_batch_size:{args.unfreezing_batch_size}\nfreezing_epoch_size:{args.freezing_batch_size}")
    logger.debug(f"\n")
    logger.debug(f"The details of optimaizer：\noptimizer:{args.optimizer_type}\ninit_lr:{args.init_lr}\nmin_lr:{args.min_lr}")
    logger.debug(f"\n")

    writer = SummaryWriter(log_dir=os.path.join(time_str,"tensorbord_log"))
    input = torch.from_numpy(np.random.rand(1,3,args.image_shape[0],args.image_shape[1])).type(torch.FloatTensor)
    if args.cuda:
        input=input.cuda()

    writer.add_graph(model,input)


    input = torch.randn(1, 3, args.image_shape[0], args.image_shape[1]).cuda()
    flops, params = profile(model, inputs=(input,))
    logger.debug(
       f"The details of the model：\nflops:{flops/1000000}M\nparams:{params/1000000}M\n")
    logger.debug(f"\n")





    best_val_loss=1000
    for epoch in range(args.epoch):

        if args.freezing_flag and args.freezing_epoch:
            if epoch<args.freezing_epoch:
                train_loader = DataLoader(train_dataset, pin_memory=True, shuffle=True, sampler=None,
                                          batch_size=args.freezing_batch_size,
                                          drop_last=True, collate_fn=centernet_dataset_collate,
                                          num_workers=args.num_workes)

                val_loader = DataLoader(val_dataset, pin_memory=True, shuffle=True, sampler=None,
                                        batch_size=args.freezing_batch_size,
                                        drop_last=True, collate_fn=centernet_dataset_collate,
                                        num_workers=args.num_workes)

                model.freeze_backbone()
                optimizer, scheduler_lr = lr_set(args.init_lr, args.min_lr, args.freezing_batch_size, args.optimizer_type,
                                                 model=model, weight_decay=args.weight_decay, epoch=args.epoch)

                epoch_step_train = num_trainset // args.freezing_batch_size
                epoch_step_val = num_testset // args.freezing_batch_size
            else:
                train_loader = DataLoader(train_dataset, pin_memory=True, shuffle=True, sampler=None,
                                          batch_size=args.unfreezing_batch_size,
                                          drop_last=True, collate_fn=centernet_dataset_collate,
                                          num_workers=args.num_workes)

                val_loader = DataLoader(val_dataset, pin_memory=True, shuffle=True, sampler=None,
                                        batch_size=args.unfreezing_batch_size,
                                        drop_last=True, collate_fn=centernet_dataset_collate,
                                        num_workers=args.num_workes)
                model.unfreeze_backbone()
                optimizer, scheduler_lr = lr_set(args.init_lr, args.min_lr, args.unfreezing_batch_size,
                                                 args.optimizer_type,
                                                 model=model, weight_decay=args.weight_decay, epoch=args.epoch)

                epoch_step_train = num_trainset // args.unfreezing_batch_size
                epoch_step_val = num_testset // args.unfreezing_batch_size

        else:
            train_loader = DataLoader(train_dataset, pin_memory=True, shuffle=True, sampler=None,
                                      batch_size=args.unfreezing_batch_size,
                                      drop_last=True, collate_fn=centernet_dataset_collate,
                                      num_workers=args.num_workes)

            val_loader = DataLoader(val_dataset, pin_memory=True, shuffle=True, sampler=None,
                                    batch_size=args.unfreezing_batch_size,
                                    drop_last=True, collate_fn=centernet_dataset_collate,
                                    num_workers=args.num_workes)
            model.unfreeze_backbone()
            optimizer, scheduler_lr = lr_set(args.init_lr, args.min_lr, args.unfreezing_batch_size,
                                             args.optimizer_type,
                                             model=model, weight_decay=args.weight_decay, epoch=args.epoch)

            epoch_step_train = num_trainset // args.unfreezing_batch_size
            epoch_step_val = num_testset // args.unfreezing_batch_size


        print('Epoch:' + str(epoch + 1) + '/' + str(args.epoch))
        set_optimizer_lr(optimizer, scheduler_lr, epoch)
        print(optimizer.param_groups[0]['lr'])
        trainval_one_epoch(train_datasets=train_loader,val_datasets=val_loader,model=model,apm=args.amp,
                                   optimizer=optimizer,writer=writer,epoch=epoch,iter_num_train=epoch_step_train,
                                   iter_num_val=epoch_step_val,scaler=scaler,cuda=args.cuda,
                                    best_val_loss=best_val_loss,weigth_path = args.weight_save_path)





if __name__=="__main__":
    train()























