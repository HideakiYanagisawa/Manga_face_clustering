import os
import numpy as np
import glob
import shutil
from PIL import Image, ImageFilter
import xml.etree.ElementTree as ET

if __name__ == '__main__':
    image_set_file = 'finetune.txt'
    data_path = '/home/dl-box/fine-tune/VOC2012' # Dataset folder
    images = [] # images
    classes = [] # character numbers 
    labels = [] # label of character images
    num = 0
    with open(image_set_file) as f:
        file_index = [x.strip() for x in f.readlines()]
    file_path = './train_imges'
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    for index in file_index:
        print index
        tree = ET.parse(data_path + '/Annotations/' + index + '.xml')
        root = tree.getroot()
        classes_title = []
        labels_title = []
        for page in root[1]:
            for obj in page.findall('face'):
                cls = obj.get('character')
                x1 = int(obj.get('xmin'))
                y1 = int(obj.get('ymin'))
                x2 = int(obj.get('xmax'))
                y2 = int(obj.get('ymax'))
                bbox_width = x2-x1
                bbox_height = y2-y1
                if bbox_width >=30 and bbox_height >=30:
                    if cls not in classes_title:
                        classes_title.append(cls)
                    labels_title.append(classes_title.index(cls))
        for i in range(len(classes_title)):
            if labels_title.count(i) >= 10:
                classes.append(classes_title[i])
                os.mkdir(file_path + '/' + classes_title[i])

        for page in root[1]:
            num_ind = int(page.get('index'))
            image_index = data_path + '/images/' + index + '/' + '{0:03d}'.format(num_ind) + '.jpg'
            im = Image.open(image_index)
            width = im.size[0]
            height = im.size[1]
            for obj in page.findall('face'):
                 cls = obj.get('character')
                 if cls in classes:
                    labels.append(classes.index(cls))
                    x1 = int(obj.get('xmin'))
                    y1 = int(obj.get('ymin'))
                    x2 = int(obj.get('xmax'))
                    y2 = int(obj.get('ymax'))
                    bbox_width = x2-x1
                    bbox_height = y2-y1
                    if bbox_width >=30 and bbox_height >=30:
                        re_x1 = int(max(0, x1 - bbox_width/2))
                        re_y1 = int(max(0, y1 - bbox_height/2))
                        re_x2 = int(min(width, x2 + bbox_width/2))
                        re_y2 = int(min(height, y2 + bbox_height/2))
                        im_bbox = im.crop((re_x1, re_y1, re_x2, re_y2))
                        im_bbox.save(file_path + '/' + cls + '/' + '{0:06d}'.format(num) + '.jpg', 'jpeg')
                        num += 1


