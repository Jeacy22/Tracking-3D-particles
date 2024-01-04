# -----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# -----------------------------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from CenterNet_test import CenterNet

if __name__ == "__main__":
    centernet = CenterNet()
    mode = "predict"

    crop = False
    count = False
    video_path = 0
    video_save_path = ""
    video_fps = 25.0

    test_interval = 100
    fps_image_path = ""

    dir_origin_path = ""
    dir_save_path = ""
    heatmap_save_path = "output/heatmap_vision.png"
    imagepath = 'output/original_image.png'
    simplify = True
    onnx_save_path = "model_data/models.onnx"

    if mode == "predict":
        import os

        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = centernet.detect_image(image, crop=crop, count=count)
                r_image.save(os.path.join(dir_save_path, "result1.png"))
                r_image.show()

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = centernet.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os
        import time
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        img_names =sorted(img_names)
        info=list()
        info_amount = list()
        start = time.time()
        for i, img_name in enumerate(tqdm(img_names)):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                boxes,depth,label ,len1= centernet.detect_image2(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                #r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")))
                depth = np.expand_dims(depth, axis=1)
                label = np.expand_dims(label, axis=1)
                all = np.concatenate((boxes,depth,label),axis = 1)

                all = all.tolist()
                info.append(all)
                info_amount.append(len1)
        b = []
        for i in range(len(info)):
            b.append(np.array(info[i]))
        c = np.vstack(b)
        end = time.time()
        print(str(end))
        np.savetxt(os.path.join(dir_save_path, "result.csv"), c,delimiter=",")
        np.savetxt(os.path.join(dir_save_path, "result_index.csv"), info_amount, delimiter=",")


    elif mode == "heatmap":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                centernet.detect_heatmap(image, heatmap_save_path,imagepath)

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
