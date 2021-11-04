import os
import coord_io
import readimage
import cv2
import numpy as np
import torch
import torch.nn as nn
import multiprocessing
import warnings
from scipy import signal
from readimage import image_read, image_write
import matplotlib.pyplot as plt


def gaussian_filter(sigma, s=11):
    dim = s // 2
    xx, yy = np.meshgrid(np.arange(-dim, dim + 1), np.arange(-dim, dim + 1))
    d = xx ** 2 + yy ** 2
    f = np.exp(-0.5 * d / sigma ** 2)
    return f


def lowfilter(image):
    b, a = signal.butter(8, 0.2, 'lowpass')
    denoised = signal.filtfilt(b, a, image)  # data为要过滤的信号
    _, ax = plt.subplots(1, 2, figsize=(24, 12))
    ax[0].imshow(image, cmap='gray')
    ax[1].imshow(denoised, cmap='Greys_r')
    plt.savefig("denoise_try0.2.png")
    return denoised


def equal_hist(image_path, output_path):
    image = image_read(image_path)
    gray = (image).astype(np.uint8)
    gray = np.array(gray)
    equal_hist_image = cv2.equalizeHist(gray)
    # _, ax = plt.subplots(1, 2, figsize=(24, 12))
    # ax[0].imshow(image, cmap='Greys_r')
    # ax[1].imshow(equal_hist_image, cmap='Greys_r')
    # plt.imshow(equal_hist_image, cmap='Greys_r')
    # save_compare_path = os.path.join(output_path, os.path.splitext(os.path.basename(image_path))[0])+'compare.jpg'
    save_preprocessed_path = os.path.join(output_path, os.path.splitext(os.path.basename(image_path))[0]) + '.jpg'
    # plt.savefig(save_preprocessed_path)
    cv2.imwrite(save_preprocessed_path, equal_hist_image)

    return equal_hist_image


def preprocess(image_path, output_path,  is_equal_hist=True, denoise_model=None):
    if is_equal_hist:
        equal_hist(image_path, output_path)

    if denoise_model:
        if denoise_model:
            if denoise_model == 'lowpass':
                lowfilter(image)

            # elif denoise_model=='n2n':
            # TODO: other denoise models


def preprocess_images(coco_dir, has_annots=True):
    pwd = os.getcwd()
    if has_annots:
        # 获取项目文件夹的位置
        root_dir = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
        output_annot_dir = os.path.join(coco_dir, "preprocessed_annots")
        output_images_dir = os.path.join(coco_dir, "preprocessed_images")
        if not os.path.exists(output_annot_dir):
            os.makedirs(output_annot_dir)
        if not os.path.exists(output_images_dir):
            os.makedirs(output_images_dir)
        print(coco_dir)

        for sub_dir in os.listdir(coco_dir):
            if os.path.isdir(coco_dir + sub_dir) and not sub_dir.startswith("pre"):
                print("--------------------------------------------------")
                print(coco_dir + sub_dir)
                ann_dir = os.path.join(coco_dir + sub_dir, "annotations")
                print(f"cp {ann_dir}/* {output_annot_dir}")
                os.system(f"cp {ann_dir}/* {output_annot_dir}")
                img_dir = os.path.join(coco_dir + sub_dir, "micrographs")
                print("image_dir: ", img_dir)
                # img_annot_pairs = find_image_annot_pairs_by_dir(ann_dir, img_dir)
                # imgs, annots = annotation_parser(img_annot_pairs, box_size=220)
                # print(len(imgs))

                for img in os.listdir(img_dir):
                    image_path = os.path.join(img_dir, img)
                    if image_path.endswith("mrc"):
                        equal_hist_image = equal_hist(image_path, output_images_dir)

    else:
        # 获取项目文件夹的位置
        root_dir = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
        coco_dir = os.path.join(root_dir, "coco/coco_split/test/")
        output_images_dir = os.path.join(coco_dir, "preprocessed_images")
        print(output_images_dir)
        if not os.path.exists(output_images_dir):
            os.makedirs(output_images_dir)

        for sub_dir in os.listdir(coco_dir):
            if sub_dir.endswith(".jpg"):
                print("--------------------------------------------------")
                print(coco_dir + sub_dir)
                img_dir = os.path.join(coco_dir + sub_dir)
                print("image_dir: ", img_dir)
                equal_hist_image = equal_hist(img_dir, output_images_dir)


def find_image_annot_pairs(annotations, images):
    import difflib
    img_names = list(map(os.path.basename, images))
    img_anno_pairs = []
    for ann in annotations:
        ann_without_ext = os.path.splitext(os.path.basename(ann))[0]
        cand_list = [i for i in img_names if ann_without_ext in i]
        try:
            cand_list_no_ext = list(map(os.path.basename, cand_list))
            corresponding_img_path = difflib.get_close_matches(ann_without_ext, cand_list_no_ext, n=1, cutoff=0)[0]
            corresponding_img_path = cand_list[cand_list_no_ext.index(corresponding_img_path)]
        except IndexError:
            print("Cannot find corresponding image file for ", ann, '- Skipped.')
            continue
        index_image = img_names.index(corresponding_img_path)
        img_anno_pairs.append((images[index_image], ann))
    return img_anno_pairs


def find_image_annot_pairs_by_dir(ann_dir, img_dir):
    if not os.path.exists(ann_dir):
        import sys
        print("Annotation folder does not exist:", ann_dir, "Please check your config file.")
        sys.exit(1)

    if not os.path.exists(img_dir):
        import sys
        print("Your image folder does not exist:", ann_dir, "Please check your config file.")
        sys.exit(1)

    # Scan all image filenames
    img_files = []
    for root, directories, filenames in os.walk(img_dir, followlinks=True):
        for filename in filenames:
            if filename.endswith(
                    ("jpg", "png", "mrc", "mrcs", "tif", "tiff")
            ) and not filename.startswith("."):
                img_files.append(os.path.join(root, filename))

    # Read annotations
    annotations = []
    for root, directories, filenames in os.walk(ann_dir, followlinks=True):
        for ann in sorted(filenames):
            if ann.endswith(("star", "box", "txt")) and not filename.startswith("."):
                annotations.append(os.path.join(root, ann))
    img_annot_pairs = find_image_annot_pairs(annotations, img_files)

    return img_annot_pairs


if __name__ == "__main__":
    # root_dir = '/home/zhangchi/Deformable-DETR/data/coco10028tiny/'
    root_dir = '/home/zhangchi/cryodata/empiar10406/'

    image_dir = root_dir + 'train/'
    output_dir = root_dir + 'preprocess_train/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for image in os.listdir(image_dir):
        print(f'preprocess {image} now...')
        preprocess(image_dir+image, output_dir)
