"""
对原始micrograph图像训练数据进行裁切，生成固定大小的patches，适用于HBB(Horizontal Bounding Box)
"""
import cv2
import os
import csv
import numpy as np
import glob
from coord_io import read_eman_boxfile, read_star_file, write_star_file
from read_image import image_read, image_write
import mrcfile

def iou(BBGT, imgRect):
    """
    计算每个BBGT和图像块所在矩形区域的交与BBGT本身的的面积之比，比值范围：0~1
    输入：BBGT：n个标注框，大小为n*4,每个标注框表示为[xmin,ymin,xmax,ymax]，类型为np.array
          imgRect：裁剪的图像块在原图上的位置，表示为[xmin,ymin,xmax,ymax]，类型为np.array
    返回：每个标注框与图像块的iou（并不是真正的iou），返回大小n,类型为np.array
    """
    left_top = np.maximum(BBGT[:, :2], imgRect[:2])
    right_bottom = np.minimum(BBGT[:, 2:], imgRect[2:])
    wh = np.maximum(right_bottom - left_top, 0)
    inter_area = wh[:, 0] * wh[:, 1]
    iou = inter_area / ((BBGT[:, 2] - BBGT[:, 0]) * (BBGT[:, 3] - BBGT[:, 1]))
    return iou


def split(imgname, dirsrc, dirdst, split_num=2, gap=100, iou_thresh=0.3, ext='.mrc', is_preprocessed=False):
    """
    split images with annotation files.
    imgname:   待裁切图像名（带扩展名）
    dirsrc:    待裁切的图像保存目录的上一个目录，默认图像与标注文件在一个文件夹下，图像在images下，标注在labelTxt下，标注文件格式为每行一个gt,
               格式为xmin,ymin,xmax,ymax,class
    dirdst:    裁切的图像保存目录的上一个目录，目录下有images,labelTxt两个目录分别保存裁切好的图像或者txt文件，
               保存的图像和txt文件名格式为 oriname_min_ymin.png(.txt),(xmin,ymin)为裁切图像在原图上的左上点坐标,txt格式和原文件格式相同
    subsize:   裁切图像的尺寸，默认为正方形
    gap:       相邻行或列的图像重叠的宽度，默认设置成Bbox的宽度
    iou_thresh:小于该阈值的BBGT不会保存在对应图像的txt中（在图像过于边缘或与图像无交集）
    ext:       保存图像的格式，默认为mrc
    """
    if is_preprocessed:
        path = os.path.join(os.path.join(dirsrc, 'preprocessed/'), imgname)
    else:
        path = os.path.join(os.path.join(dirsrc, 'micrographs/'), imgname)
    print(path)

    img = image_read(path)
    img_h, img_w = img.shape[:2]
    subsize_h, subsize_w = img_h // int(split_num) + gap, img_w // int(split_num) + gap
    BBGT = []

    # read box file
    box_file_path = os.path.join(dirsrc, 'annots/') + imgname[:-4] + '.box'
    if os.path.exists(box_file_path):
        boxes = read_eman_boxfile(box_file_path)
    # read star file
    box_file_path = os.path.join(dirsrc, 'annots/') + imgname[:-4] + '.star'
    if os.path.exists(box_file_path):
        boxes = read_star_file(box_file_path, box_width=gap)

    for box in boxes:
        box_width = int(box.w)
        box_height = int(box.h)
        box_xmin = int(box.x)
        box_ymin = img_h - (int(box.y) + box_height)
        box_xmax = box_xmin + box_width
        box_ymax = box_ymin + box_height
        BBGT.append([box_xmin, box_ymin, box_xmax, box_ymax])
        # BBGT.append([box_xmin, img_h - (box_ymin + box.h), box.w, box.h])  # box.x, box.y, box.w, box.h
    BBGT = np.array(BBGT)

    top = 0
    reachbottom = False
    while not reachbottom:
        reachright = False
        left = 0
        if top + subsize_h >= img_h:
            reachbottom = True
            top = max(img_h - subsize_h, 0)
        while not reachright:
            if left + subsize_w >= img_w:
                reachright = True
                left = max(img_w - subsize_w, 0)
            imgsplit = img[top:min(top + subsize_h, img_h), left:min(left + subsize_w, img_w)]
            if imgsplit.shape[:2] != (subsize_h, subsize_w):
                template = np.zeros((subsize_h, subsize_w, 3), dtype=np.uint8)
                template[0:imgsplit.shape[0], 0:imgsplit.shape[1]] = imgsplit
                imgsplit = template

            print(imgsplit)

            if not np.issubdtype(imgsplit.dtype, np.float32):
                imgsplit = imgsplit.astype(np.float32)

            mean = np.mean(imgsplit)
            sd = np.std(imgsplit)

            imgsplit = (imgsplit - mean) / sd
            imgsplit[imgsplit > 3] = 3
            imgsplit[imgsplit < -3] = -3

            image_write(os.path.join(os.path.join(dirdst, 'micrographs'),
                                     imgname.split('.')[0] + '_' + str(left) + '_' + str(top) + '.mrc'), imgsplit)

            imgrect = np.array([left, top, left + subsize_w, top + subsize_h]).astype('float32')
            ious = iou(BBGT[:, :4].astype('float32'), imgrect)
            BBpatch = BBGT[ious > iou_thresh]
            print("bbox number： ", len(BBpatch))

            path = os.path.join(os.path.join(dirdst, 'annots'),
                                imgname.split('.')[0] + '_' + str(left) + '_' + str(top) + '.box')

            with open(path, "w") as boxfile:
                boxwriter = csv.writer(
                    boxfile, delimiter='\t', quotechar="|", quoting=csv.QUOTE_NONE
                )
                for bb in BBpatch:  # [box_xmin, box_ymin, box_xmax, box_ymax]
                    boxheight = bb[3] - bb[1]
                    boxwidth = bb[2] - bb[0]
                    xmin = int(bb[0]) - left
                    ymin = int(bb[1]) - top
                    xmax = int(bb[2]) - left
                    ymax = int(bb[3]) - top
                    # [box.x, box.y, box.w, box.h], box.x, box,y = lower left corner
                    boxwriter.writerow([xmin, subsize_h - (ymin + boxheight), boxwidth, boxheight])
            left += subsize_w - gap
        top += subsize_h - gap


def split_only_images(imgname, dirsrc, dirdst, split_num=2, gap=200, ext='.mrc'):
    img = cv2.imread(os.path.join(dirsrc, imgname), -1)
    img_h, img_w = img.shape[:2]
    print(imgname, img_w, img_h)
    subsize_h, subsize_w = img_h // int(split_num) + gap, img_w // int(split_num) + gap

    top = 0
    reachbottom = False
    while not reachbottom:
        reachright = False
        left = 0
        if top + subsize_h >= img_h:
            reachbottom = True
            top = max(img_h - subsize_h, 0)
        while not reachright:
            if left + subsize_w >= img_w:
                reachright = True
                left = max(img_w - subsize_w, 0)
            imgsplit = img[top:min(top + subsize_h, img_h), left:min(left + subsize_w, img_w)]
            if imgsplit.shape[:2] != (subsize_h, subsize_w):
                template = np.zeros((subsize_h, subsize_w, 3), dtype=np.uint8)
                template[0:imgsplit.shape[0], 0:imgsplit.shape[1]] = imgsplit
                imgsplit = template

            cv2.imwrite(os.path.join(dirdst, imgname.split('.')[0] + '_' + str(left) + '_' + str(top) + ext), imgsplit)
            left += subsize_w - gap
        top += subsize_h - gap


def split_train_val_images(dirsrc, dirdst, split_num=2, gap=200, iou_thresh=0.4, ext='.mrc', is_preprocessed=False):
    """
    split images with annotation files.
    """
    if not os.path.exists(dirdst):
        os.mkdir(dirdst)
    if not os.path.exists(os.path.join(dirdst, 'micrographs')):
        os.mkdir(os.path.join(dirdst, 'micrographs'))
    if not os.path.exists(os.path.join(dirdst, 'annots')):
        os.mkdir(os.path.join(dirdst, 'annots'))
    if is_preprocessed:
        imglist = glob.glob(f'{dirsrc}/preprocessed/*.mrc')
    else:
        imglist = glob.glob(f'{dirsrc}/micrographs/*.mrc')

    imgnameList = [os.path.split(imgpath)[-1] for imgpath in imglist]

    for imgname in imgnameList:
        if imgname.endswith("mrc"):
            split(imgname, dirsrc, dirdst, split_num, gap, iou_thresh, ext, is_preprocessed)


def split_test_images():
    """
    split test images without annotation files.
    """
    if not os.path.exists(dirdst):
        os.mkdir(dirdst)

    imglist = glob.glob(f'{dirsrc}/*.mrc')
    imgnameList = [os.path.split(imgpath)[-1] for imgpath in imglist]
    for imgname in imgnameList:
        if imgname.endswith("mrc"):
            split_only_images(imgname, dirsrc, dirdst, split_num, gap, ext)


def main():
    # split train and val images
    split_num = 2
    gap = 200
    ext = '.mrc'
    iou_thresh=0.4
    is_preprocessed=False
    dirsrc = './data/empiar10028'  # 待裁剪图像所在目录的上级目录
    dirdst = './data/empiar10028/split'  # 裁剪结果存放目录，格式和原图像目录一样
    split_train_val_images(dirsrc, dirdst, split_num, gap, iou_thresh, ext, is_preprocessed)
    
    # split test images
    # dirsrc = './data/empiar10028/test'
    # dirdst = './data/empiar10028/test_split/'
    # split_test_images(dirsrc, dirdst, split_num, gap, ext)


if __name__ == '__main__':
    main()