import numpy as np
import cv2
import os
from glob import glob
from tqdm import tqdm , trange
import random

def calculate(sample):
    "calculate the mean adn std of RGB channels respectively"
    ":param sample the sample of orginal dataset"

    means, stdevs = [], []
    img_list = []
    img_h, img_w = 40, 40  # set depends on your dataset

    for single_img_path in tqdm(sample,colour='green'):
        img = cv2.imread(single_img_path)
        img = cv2.resize(img, (img_w, img_h))
        img = img[:, :, :, np.newaxis]
        img_list.append(img)

    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))


    means.reverse()
    stdevs.reverse()
    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    return means,stdevs

if __name__ == "__main__":
    ave_mean , ave_std = [0.,0.,0.] , [0.,0.,0.]
    n = 6
    sample_num = 1000
    TRAIN_DATASET_PATH = 'F:\\3D_track\\data_big\\data4\\PNGimage'
    image_fns = glob(
        os.path.join(TRAIN_DATASET_PATH, '**.png'))  # get the address of all of the images in target folder and subfolder
    print('The number of the dataset: ', len(image_fns))

    for i in trange(n):
        sample = random.sample(image_fns,sample_num) #随机抽样
        mean , stdev = calculate(sample)
        ave_mean = np.sum([ave_mean,mean],axis=0)
        ave_std = np.sum([ave_std, stdev], axis=0)

    print("ave_normMean = {}".format(ave_mean[:] / n))
    print("ave_normStd = {}".format(ave_std[:] / n))




