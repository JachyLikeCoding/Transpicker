"""
Remove the false positive results on ice and carbon region
"""
import os

import micrograph_cleaner_em as mce
import cv2
import mrcfile
import numpy as np
import os
from transpicker.read_image import read_mrc, image_read
import matplotlib.pyplot as plt


def clean_micrograph(image_path, boxsize=200):
    # read image
    micrograph = image_read(image_path)

    # with mrcfile.open(image_path, permissive=True, mode='r') as mrc:
    #     mic = mrc.data

    # By default, the mask predictor will try load the model at
    # "~/.local/share/micrograph_cleaner_em/models/"
    # provide , deepLearningModelFname= modelPath argument to the builder
    # if the model is placed in other location
    # TODO: model的路径、 boxsize 、GPU 均需要改成用户自己输入
    deepLearningModelFname = "/home/zhangchi/.local/share/micrograph_cleaner_em/models/defaultModel.keras"
    with mce.MaskPredictor(boxsize, deepLearningModelFname=deepLearningModelFname, gpus=[0]) as mp:
        mask = mp.predictMask(micrograph)  # by default, mask is float32 numpy array

    print("mask:", mask)
    return mask


def save_mask(image_path, mask):
    name_without_ext = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.dirname(image_path)
    save_path = save_path + '/mask/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    mask_name_mrc = save_path + name_without_ext + '_mask.mrc'
    mask_name_jpg = save_path + name_without_ext + '_mask.jpg'
    mask_name_txt = save_path + name_without_ext + '_mask.txt'

    # Then write the mask as a txt file, mrc file and jpg file
    # np.savetxt(mask_name_txt, mask)

    with mrcfile.new(mask_name_mrc, overwrite=True) as maskFile:
        maskFile.set_data(mask.astype(np.half))  # as float

    import cv2
    cv2.imwrite(mask_name_jpg, mask * 255)

    print('mask:', mask)


def display_compare(name):
    name_without_ext = os.path.splitext(os.path.basename(name))[0]
    mask_name = "/home/zhangchi/Deformable/result0628/mask/" + name_without_ext + '_mask.jpg'
    print('mask name: ', mask_name)
    image_name = "/home/zhangchi/Deformable/data/cocochier/test/" + name_without_ext + '.jpg'
    original = image_read(name)
    mask = image_read(mask_name)

    # display and save compare results.
    plt.subplots(1, 2)
    plt.subplot(1, 2, 1)
    plt.imshow(original, 'gray')
    plt.title('original')
    plt.subplot(1, 2, 2)
    plt.imshow(mask, 'gray')
    plt.title('cleaned')
    plt.savefig("/home/zhangchi/Deformable-DETR/result0628/mask/" + name_without_ext + "_compare.png")
    plt.show()


if __name__ == "__main__":
    boxsize = 200
    image_path = "/home/zhangchi/detr/cryococo/10028/micrographs/"
    for image in os.listdir(image_path):
        image = image_path + image
        # if os.path.isfile(image_path+image):
        if image.endswith(".mrc"):
            mask = clean_micrograph(image, boxsize)
            save_mask(image, mask)
            display_compare(image)
