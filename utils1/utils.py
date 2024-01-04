import numpy as np
from PIL import Image
from functools import partial
import math




def cvtColor(image):
    if len(np.shape(image)) ==3 and np.shape(image)[2]==3:
        return image
    else:
       image = image.convert('RGB')
       return image

def preprocess_input(image):
    image   = np.array(image,dtype = np.float32)[:, :, ::-1]
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(0, 0.5 ** 0.5, image.shape)
    image = image+noise

    if image.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.

    image = np.clip(image, low_clip, 1.0)
    image = np.uint8(image * 255)

    mean    = [0.49397988, 0.49397988, 0.49397988]
    std     = [0.0493484, 0.0493484, 0.0493484]
    return (image / 255. - mean) / std

def preprocess_input_eval(image):
    image = np.array(image, dtype=np.float32)[:, :, ::-1]
    mean    = [0.49397988, 0.49397988, 0.49397988]
    std     = [0.0493484, 0.0493484, 0.0493484]
    image = np.uint8(image * 255)
    return (image / 255. - mean) / std





def get_classes(classes_path):
    with open(classes_path,encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names,len(class_names)


def download_weights(backbone, model_dir="./pretrained_train"):
    import os
    from torch.hub import load_state_dict_from_url


    download_urls = {
        'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    }
    url = download_urls[backbone]

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)

def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    scale = min(w / iw, h / ih)
    gw  = float(40*scale)
    gh = float(40*scale)
    if letterbox_image:

        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image,gw,gh

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image










