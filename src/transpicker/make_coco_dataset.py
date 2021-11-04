import os
import csv
import cv2
import json
import argparse
import numpy as np
from utils import BoundBox
from read_image import image_read, read_width_height, image_write
from coord_io import read_eman_boxfile, read_star_file


def get_args_parser():
    parser = argparse.ArgumentParser('TransPicker', add_help=False)
    
    parser.add_argument('--coco_path', default='/home/zhangchi/transpicker/Transpicker/data/empiar10028/', type=str)
    parser.add_argument('--images_path', default='/home/zhangchi/transpicker/Transpicker/data/empiar10028/micrographs/', type=str)
    parser.add_argument('--phase', default='train', type=str)
    parser.add_argument('--box_width', default=200, type=int)
    return parser


def make_coco_dataset(root_path, image_path, box_width=200, phase='train'):
    """Make coco-style dataset. """

    if not os.path.exists(os.path.join(root_path, phase)):
        os.makedirs(os.path.join(root_path, phase))

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
    print(f"The No.{split} is the split number.")

    for index in indexes:
        if index.startswith(".") or os.path.isdir(image_path + index):
            indexes.remove(index)

    if phase == 'train':
        indexes = [line for i, line in enumerate(indexes) if i <= split]
    elif phase == 'val':
        indexes = [line for i, line in enumerate(indexes) if i > split]

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

            image_write(f'{root_path}/{phase}/{index[:-4]}.jpg', image)
        elif index.endswith(('.jpg', '.png')):
            os.system(f"cp {image_path}/{index} {root_path}/{phase}/")
        else:
            raise Exception(f"{image_path}/{index} is not supported image format.")

    anno_id = 0

    # read image width and height , read bounding boxes
    for k, index in enumerate(indexes):
        width, height = read_width_height(os.path.join(image_path) + index)
        print("width:", width, "   height:", height)
        # 添加图像的信息到dataset中
        dataset['images'].append({'file_name': index[:-4] + '.jpg',  # 这里构建的是jpg数据集
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
            # read star file
            box_file_path = os.path.join(root_path, 'annots/') + index[:-4] + '.star'

            if os.path.exists(box_file_path):
                boxes = read_star_file(box_file_path, box_width=box_width)

        # TODO: READ OTHER FILE TYPES.

        for box in boxes:
            box_xmin = int(box.x)
            box_width = int(box.w)
            box_height = int(box.h)
            box_ymin = height - (int(box.y) + box_height)
            # box_xmax = box_xmin + box_width
            # box_ymax = box_ymin + box_height

            anno_id += 1
            dataset['annotations'].append({
                'area': box.w * box.h,
                'bbox': [box_xmin, box_ymin, box.w, box.h],
                'category_id': 1,  # particle
                'id': anno_id,
                'image_id': k,
                'iscrowd': 0,
                'segmentation': []
            })

    # Folder to save the results
    folder = os.path.join(root_path, 'annotations')
    if not os.path.exists(folder):
        os.makedirs(folder)
    json_name = os.path.join(root_path, 'annotations/instances_{}.json'.format(phase))
    print("json_name:", json_name)
    with open(json_name, 'w') as f:
        json.dump(dataset, f)


def make_test_dataset(dataset_path):
    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')
    test_path = os.path.join(dataset_path, 'test')
    micrographs_path = os.path.join(dataset_path, 'micrographs')

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    train_files = os.listdir(train_path)
    val_files = os.listdir(val_path)
    all_files = os.listdir(micrographs_path)

    for file in all_files:
        jpgfile = file[:-4]+'.jpg'
        if (jpgfile not in train_files) and (jpgfile not in val_files):
            image = image_read(f'{micrographs_path}/{file}')
            if not np.issubdtype(image.dtype, np.float32):
                image = image.astype(np.float32)
            mean = np.mean(image)
            sd = np.std(image)
            image = (image - mean) / sd
            image[image > 3] = 3
            image[image < -3] = -3

            image_write(f'{test_path}/{file[:-4]}.jpg', image)


def main(args):
    print(args)
    make_coco_dataset(args.coco_path, args.images_path, box_width=args.box_width, phase=args.phase)
    make_coco_dataset(args.coco_path, args.images_path, box_width=args.box_width, phase='val')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Cryococo dataset preperation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

    # python make_coco_dataset.py