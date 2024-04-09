import os.path
import xml.etree.ElementTree as ET
from utils import get_config
import os
import numpy as np
from pathlib import Path
import re
from utils import get_logger




def open_file(path):
    with open(path) as f:
        data_infos = f.readlines()
        if data_infos[-1]=="":
            del data_infos[-1]
    return  len(data_infos),data_infos

def read_xml(path,classes,list_file,nums):
    tree = ET.parse(path)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('different') != None:
            difficult = obj.find('different').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(float(xmlbox.find('xmin').text)), float(float(xmlbox.find('ymin').text)),
             float(float(xmlbox.find('xmax').text)), float(float(xmlbox.find('ymax').text)),
             float(float(xmlbox.find('deptho').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        nums[classes.index(cls)] = nums[classes.index(cls)] + 1

    return nums

def get_data_txt(path_index,path_annotation,type,path_image,classes,config_path):
    index_path = os.path.join(path_index,type+".txt")
    infos_txtx_path = f"{type}.txt"
    if os.path.exists(infos_txtx_path):
        os.remove(infos_txtx_path)
    if not os.path.exists(index_path):
        return None,None

    with open(os.path.join(config_path,"config.txt"),"r") as f:
       datalines = f.readlines()
    deleteindex=0

    for i,line in enumerate(datalines):
      if re.match(f"{type}_dataset",line):
          deleteindex = i
    if deleteindex!=0:
      del datalines[deleteindex]

    with open(os.path.join(config_path, "config.txt"), "w") as f:
       info = f"{type}_dataset={os.path.join(os.getcwd(),infos_txtx_path)}\n"
       datalines.append(info)
       f.writelines(datalines)



    num_data,data_index = open_file(index_path)
    nums = np.zeros(len(classes))
    for info in data_index:
        info=info.replace("\n",'')
        path_xml = os.path.join(path_annotation,info+".xml")
        path_singleimage = os.path.join(path_image,info+".png")
        if not os.path.exists(path_singleimage) or not os.path.exists(path_xml):
            return None,None

        with open(infos_txtx_path,'a+') as f:
            f.write(path_singleimage)
            num_type=read_xml(path_xml, classes, f,nums)
            f.write('\n')

    return num_data,num_type



if __name__=="__main__":
    current_path = Path.cwd()
    updated_path = current_path.parent
    config_dict = get_config(updated_path)
    logger =get_logger(current_path)
    annotation_path = config_dict["annotation_path"]
    split_index_path = config_dict['split_index_path']
    image_path = config_dict['img_path']
    classes=config_dict['classes'].split(',')
    datatype = config_dict['datatype'].split(',')
    if not os.path.exists(annotation_path):
        OSError("Annotation files not found")
    i=0
    for i,type in enumerate(datatype):
        num_data, num_type=get_data_txt(split_index_path,annotation_path,type,image_path,classes,updated_path)
        if num_data==None:
            OSError("files do not find")
        i=i+1
        logger.debug(f"the amount of {type} images is  {num_data}")
        logger.debug(f"the amount of the {classes[0]} particle is {num_type[0]}, and the amount of the {classes[1]} particle is {num_type[1]}")
        logger.debug(f"+++++++++++++++++++++++++++++++++++++++")





















