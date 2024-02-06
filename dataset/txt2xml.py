import time
import os
from PIL import Image
import cv2
import numpy as np


out0 = '''<annotation>
    <folder>%(folder)s</folder>
    <filename>%(name)s</filename>
    <path>%(path)s</path>
    <source>
        <database>None</database>
    </source>
    <size>
        <width>%(width)d</width>
        <height>%(height)d</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
'''
out1 = '''    <object>
        <name>%(class)s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%(xmin)d</xmin>
            <ymin>%(ymin)d</ymin>
            <xmax>%(xmax)d</xmax>
            <ymax>%(ymax)d</ymax>
            <deptho>%(deptho)f</deptho>
        </bndbox>
    </object>
'''

out2 = '''</annotation>
'''




def translate(fdir, lists,file_dir_xml):
    source = {}
    label = {}
    for png in lists:
        print(png)
        if png[-4:] == '.png':
            image = cv2.imread(png)
            h, w, d = image.shape
            fxml = file_dir_xml+png[-9:-4]+'.xml'
            fxml = open(fxml, 'w')
            imgfile = png.split('/')[-1]
            source['name'] = imgfile
            source['path'] = png
            source['folder'] = os.path.basename(fdir)

            source['width'] = w
            source['height'] = h

            fxml.write(out0 % source)

            txt = png.replace('.png', '.txt')

            lines = np.loadtxt(txt,dtype=float)


            if len(np.array(lines).shape) == 1:
                lines = [lines]

        for box in lines:
            if box.shape != (8,):
                box = lines


            if int(box[7])==0:
               label['class'] = "small"
            elif int(box[7])==1:
                label['class'] = "big"

            centerx = float(box[4])
            centery = float(box[5])
            xmin = centerx-20
            ymin = centery-20
            xmax = centerx+20
            ymax = centery+20
            label['xmin'] = xmin
            label['ymin'] = ymin
            label['xmax'] = xmax
            label['ymax'] = ymax
            label['deptho'] = box[6]
            fxml.write(out1 % label)
        fxml.write(out2)

if __name__ == '__main__':
    file_dir = ''
    file_dir_xml =''
    lists = []
    for i in os.listdir(file_dir):
        if i[-3:] == 'png':
            lists.append(file_dir + '/' + i)
    translate(file_dir, lists,file_dir_xml)
    print('---------------Done!!!--------------')

