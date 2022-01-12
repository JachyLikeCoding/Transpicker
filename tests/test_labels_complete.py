'''
Test the influence of label integrity on experimental results
'''
import os
import csv
import cv2
import json

import random
import argparse
import numpy as np
import sys
sys.path.append(os.path.dirname(sys.path[0]))
sys.path.append("/home/zhangchi/transpicker/Transpicker/src")
from transpicker.utils import BoundBox
from transpicker.read_image import image_read, read_width_height, image_write
from transpicker.coord_io import read_eman_boxfile, read_star_file


def make_coco_dataset_test_label(root_path, image_path, box_width=200, percent=100, phase='train'):
    """Make coco-style dataset. """
    if not os.path.exists(os.path.join(root_path, phase+'_percent'+str(percent))):
        os.makedirs(os.path.join(root_path, phase+'_percent'+str(percent)))

    dataset = {'categories': [], 'images': [], 'annotations': []}
    classes = ['particle']
    # Establishing the correspondence between class labels and IDs
    for i, cls in enumerate(classes, 1):
        dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})

    # Retrieve image names from the images folder
    indexes = [f for f in os.listdir(image_path)]
    print(f"There are totally {len(indexes)} micrographs in the {root_path}.")

    # split the training and testing dataset
    split = int(len(indexes) * 0.8)

    for index in indexes:
        if index.startswith(".") or os.path.isdir(image_path + index):
            indexes.remove(index)
    
    if phase == 'train':
        random.seed(10)
        sample_num = int(percent * 0.01 * split)
        indexes = [line for i, line in enumerate(indexes) if i <= split]
        indexes = random.sample(indexes, sample_num)
        print(f'percent {percent} {phase} dataset has {len(indexes)} micrographs................')
    elif phase == 'val':
        indexes = [line for i, line in enumerate(indexes) if i > split]
        print(f'percent {percent} {phase} dataset has {len(indexes)} micrographs................')

    for index in indexes:
        if index.endswith('.mrc'):
            image = image_read(f'{image_path}{index}')
            if not np.issubdtype(image.dtype, np.float32):
                image = image.astype(np.float32)
            mean = np.mean(image)
            sd = np.std(image)
            image = (image - mean) / sd
            image[image > 3] = 3
            image[image < -3] = -3
            image_write(f'{root_path}/{phase}_percent{percent}/{index[:-4]}.jpg', image)

        elif index.endswith(('.jpg', '.png')):
            os.system(f"cp {image_path}/{index} {root_path}/{phase}_percent{percent}/")

        else:
            raise Exception(f"{image_path}/{index} is not supported image format.")

    anno_id = 0

    # read image width and height , read bounding boxes
    for k, index in enumerate(indexes):
        width, height = read_width_height(os.path.join(image_path) + index)
        print("width:", width, "   height:", height)
        dataset['images'].append({'file_name': index[:-4] + '.jpg',
                                  'id': k,
                                  'width': width,
                                  'height': height})
        print(index)
        boxes = []
        # read box file or star file
        if index.endswith(("jpg", "png", "mrc")):
            box_file_path = os.path.join(root_path, 'annots/') + index[:-4] + '.box'
            if os.path.exists(box_file_path):
                boxes = read_eman_boxfile(box_file_path)
            box_file_path = os.path.join(root_path, 'annots/') + index[:-4] + '.star'
            if os.path.exists(box_file_path):
                boxes = read_star_file(box_file_path, box_width=box_width)
        # TODO: READ OTHER FILE TYPES.

        for box in boxes:
            box_width = int(box.w)
            box_height = int(box.h)
            box_xmin = int(box.x)
            box_ymin = height - (int(box.y) + box_height)

            anno_id += 1
            dataset['annotations'].append({
                'area': box.w * box.h,
                'bbox': [box_xmin, box_ymin, box.w, box.h],
                'category_id': 1,  # particle class
                'id': anno_id,
                'image_id': k,
                'iscrowd': 0,
                'segmentation': []
            })

    # Folder to save the results
    folder = os.path.join(root_path, 'annotations')
    if not os.path.exists(folder):
        os.makedirs(folder)
    json_name = os.path.join(root_path, f'annotations/instances_{phase}_percent{percent}.json')
    print("json_name:", json_name)
    with open(json_name, 'w') as f:
        json.dump(dataset, f)



if __name__ == "__main__":
    root_path = '/home/zhangchi/transpicker/Transpicker/data/empiar10028/'
    image_path = '/home/zhangchi/transpicker/Transpicker/data/empiar10028/micrographs/'
    label_percent = [20, 40, 60, 80, 100] # percent of labels to use

    for percent in label_percent:
        make_coco_dataset_test_label(root_path, image_path, box_width=200, percent=percent, phase='train')