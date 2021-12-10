"""
stitch the split images and annotations.
"""
import csv
from PIL import Image
import os
import numpy as np
import cv2
from utils import nms
from postprocess import post_processing


def stitch_image(path, out_path, prefix, gap=200, num_yx=2):
    """
        path: 图像块的路径
        out_path: 输出拼接后图像的路径
        prefix: 是图像的前缀，用来寻找它的子图们
        num_yx：定义每列有几张图像
        gap: 是分块时用的gap，在拼接的时候不能重复图像内容
    """
    if not os.path.exists(path):
        raise FileNotFoundError("The full image path is not available.")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if prefix is None:
        raise FileNotFoundError("No images to be stitch. Check your full images path.")

    filenames = sorted(os.listdir(path))
    subimages = [f for f in filenames if f.startswith(prefix)]
    print("path：", path)
    print("image nums：", len(subimages))
    print('[begin]:')
    if len(subimages) != num_yx * num_yx:
        raise ValueError("The parameters of the composite image and the requested number cannot be matched.！")

    i = 0
    list_a = []

    # *step 1:下面的for循环用于将图像合成列，只有一个参数，就是num_yx，每列有几行图像
    for subimage in subimages:
        i += 1 # i用于计数
        t = (i - 1) // num_yx # t用于换列
        im = Image.open(os.path.join(path, subimage))
        im_array = np.array(im)
        
        if (i - 1) % num_yx == 0: # 如果取的图像输入下一列的第一个，因为每列是3张图像，所以1，4，7等就是每列的第一张
            list_a.append(im_array)
        else: # 否则不是第一个数，就拼接到图像的下面
            list_a[t] = np.concatenate((list_a[t], im_array[2 * gap:, :]), axis=0)
        print(f"list_a[{t}].shape", list_a[t].shape)

    # *step 2: 合成列以后需要将列都拼接起来
    for j in range(len(list_a) - 1):
        list_a[0] = np.concatenate((list_a[0], (list_a[j + 1])[:, 2 * gap:]), axis=1)
        print(f"list_a[0].shape", list_a[0].shape)

    im_save = Image.fromarray(np.uint8(list_a[0]))
    im_save.save(out_path + prefix + "_stitch.jpg")
    print("finished")


def stitch_annotations(full_images_path, annots_path, patches_path, out_path, prefix, gap=100, num_yx=2):
    """
        annots_path: path of the patch annotations
        out_path: path of the output merged annotations
        prefix: the prefix of the image, to find the sub-images annotations
        num_yx：the number of patches for each row and column
        gap: the gap of patches
    """
    if not os.path.exists(annots_path):
        raise FileNotFoundError("The patch image annotations path is not available.")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if prefix is None:
        raise FileNotFoundError("No images to be stitch. Check your full images path.")

    filenames = sorted(os.listdir(annots_path))
    sub_annots = [f for f in filenames if f.startswith(prefix) and f.endswith('.star')]
    print("annots path：", annots_path)
    print("annots nums：", len(sub_annots))
    print('[begin]:')
    if len(sub_annots) != num_yx * num_yx:
        raise ValueError("The parameters of the composite image and the requested number cannot be matched.！")

    # 将每个子图的标注文件合并成一个文件
    nms_boxes, nms_scores = [], []
    for annot in sub_annots:
        top = annot[:-7].split('_')[-1]
        left = annot[:-4].split('_')[-2]
        top, left = int(top), int(left)
        print('top=', top, ' , left=', left)
        patch_path = patches_path + annot[:-5] + '.jpg'
        annot = annots_path + annot
        img = cv2.imread(patch_path, -1)
        img_h, img_w = img.shape[:2]

        if os.path.getsize(annot) == 0:
            print(annot, " has no bbox.")
        else:
            boxproreader = np.atleast_2d(np.genfromtxt(annot))
            boxes = [[box[0], box[1], box[2], box[3], box[4]] for box in boxproreader]  # box[4]:score
            # [xmin, subsize_h - (ymin + boxheight), boxwidth, boxheight]
            for box in boxes:
                box_width = box[2]
                box_height = box[3]
                box_xmin = int(box[0]) + left
                box_ymin = img_h - (int(box[1]) + box_height)
                box_ymin = box_ymin + top
                nms_scores.append(box[4])
                nms_box = [box_xmin, box_ymin, box_xmin + box_width, box_ymin + box_height]
                new_box = [box_xmin, img_h * num_yx - (num_yx - 1) * gap * 2 - (box_ymin + box_height), box_width, box_height]
                nms_boxes.append(nms_box)

    path = output_path + 'saved_box_files/'
    if not os.path.exists(path):
        os.mkdir(path)
    mask_path = path + prefix + '_full_addmask.box'
    path = path + prefix + '_full.box'

    nms_boxes = np.array(nms_boxes)
    print("boxes count before nms processing: ",nms_boxes.shape)

    # NMS deletes overlap boxes
    picked_boxes, picked_score = nms(nms_boxes, nms_scores, threshold=0.4)
    print("boxes count after nms processing: ", picked_boxes.shape)

    # Mask cleaner deletes boxes located in ice and carbon regions
    for f in os.listdir(full_images_path):
        if f.startswith(prefix):
            image_path = f

    masked_boxes = post_processing(picked_boxes, image_path=full_images_path+image_path, box_size=gap)
    print('boxes count after post processing', masked_boxes.shape)
    if not masked_boxes.shape == picked_boxes.shape:
        print("masked boxes exists!")

    with open(path, "w") as boxfile:
        boxwriter = csv.writer(boxfile, delimiter='\t', quotechar="|", quoting=csv.QUOTE_NONE)
        for bb in picked_boxes:
            # box_width = bb[2] - bb[0]
            # box_height = bb[3] - bb[1]
            box_width = gap
            box_height = gap
            # [box.x, box.y, box.w, box.h], box.x, box,y = lower left corner
            boxwriter.writerow([bb[0], img_h * num_yx - (num_yx - 1) * gap * 2 - (bb[1] + box_height), box_width, box_height])

    with open(mask_path,"w") as boxfile:
        boxwriter = csv.writer(boxfile, delimiter='\t', quotechar="|", quoting=csv.QUOTE_NONE)
        for bb in masked_boxes:
            box_width = gap
            box_height = gap
            # [box.x, box.y, box.w, box.h], box.x, box,y = lower left corner
            boxwriter.writerow([bb[0], img_h * num_yx - (num_yx - 1) * gap * 2 - (bb[1] + box_height), box_width, box_height])


def main():
    patches_path = './data/empiar10028/split/split_images/'
    annots_path = './outputs/empiar10028_outputs/step_star_compare/final_results/'
    full_images_path = './data/zhangchi/cryodata/empiar10096/micrographs/'
    output_path = "/data/zhangchi/transpicker_outputs/my_outputs_10096_split_giou/images_stitch/"

    # the names of full images are the prefix of patches.
    prefix_names = [prefix for prefix in os.listdir(full_images_path)
                    if os.path.isfile(os.path.join(full_images_path, prefix))]

    for p in prefix_names:
        prefix, suffix = os.path.splitext(p)
        print(prefix, suffix)  # test   .py
        stitch_image(patches_path, output_path, prefix, gap=100, num_yx=2)
        stitch_annotations(full_images_path, annots_path, patches_path, output_path, prefix, gap=100, num_yx=2)


if __name__ == '__main__':
    main()