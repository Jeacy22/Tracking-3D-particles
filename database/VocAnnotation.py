import os
import random
import xml.etree.ElementTree as ET
import numpy as np


def get_classes(classes_path):
    with open(classes_path,encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names,len(class_names)


annotation_mode = 2

classes_path = 'classes.txt'

trainval_percent = 1
train_percent = 0.7

VOCdata_path = ''
VOCdata_set = [("train"),("val")]
classes,_= get_classes(classes_path)

photo_nums = np.zeros(len(VOCdata_set))
nums = np.zeros(len(classes))

def convert_annotation(image_id,list_file):
    in_file = open(os.path.join(VOCdata_path, 'Annotation/%s.xml' % (image_id)), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('different')!=None:
            difficult = obj.find('different').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(float(xmlbox.find('xmin').text)), float(float(xmlbox.find('ymin').text)),
             float(float(xmlbox.find('xmax').text)), float(float(xmlbox.find('ymax').text)),
             float(float(xmlbox.find('deptho').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

        nums[classes.index(cls)] = nums[classes.index(cls)]+1





if __name__=="__main__":
    random.seed(0)
    if " " in os.path.abspath(VOCdata_path):
        raise ValueError("The file path cannot contain spaces.")
    if annotation_mode==0 or annotation_mode==1:
        xmlfilepath = os.path.join(VOCdata_path,'Annotation')
        saveBasePath = os.path.join(VOCdata_path,'Imageset/Main')
        temp_xml = os.listdir(xmlfilepath)
        total_xml = []
        for xml in temp_xml:
            if xml.endswith('.xml'):
                total_xml.append(xml)

        num = len(total_xml)
        list = range(num)
        tv = int(num*trainval_percent)
        tr = int(tv*train_percent)
        trainval = random.sample(list,tv)
        train = random.sample(trainval,tr)

        ftrainval = open(os.path.join(saveBasePath,'trainval.txt'),'w')
        ftest = open(os.path.join(saveBasePath,"test.txt"),'w')
        ftrain = open(os.path.join(saveBasePath,"train.txt"),'w')
        fval = open(os.path.join(saveBasePath,'val.txt'),'w')

        for i in list:
            name = total_xml[i][:-4]+'\n'
            if i in trainval:
                ftrainval.write(name)
                if i in train:
                    ftrain.write(name)
                else:
                    fval.write(name)
            else:
                ftest.write(name)

        ftrainval.close()
        ftrain.close()
        fval.close()
        ftest.close()

    if annotation_mode==0 or annotation_mode==2:
        type_index=0
        for imageset in VOCdata_set:
            image_ids = open(os.path.join(VOCdata_path, 'Imageset/Main/%s.txt'%(imageset)),
                             encoding='utf-8').read().strip().split()
            list_file = open('%s.txt' % (imageset), 'w', encoding='utf-8')

            for image_id in image_ids:
                list_file.write('%s/PNGimage/%s.png'%(os.path.abspath(VOCdata_path),image_id))
                convert_annotation(image_id,list_file)

                list_file.write('\n')
            photo_nums[type_index] = len(image_id)
            type_index += 1
            list_file.close()
        print("the amount of small particle is ",nums[0])
        print("the amount of big particle is",nums[1])
        print("the mount of val dataset is",)
        print("Generate 2007_train.txt and 2007_val.txt for train done.")














