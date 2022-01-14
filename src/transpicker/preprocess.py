import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from torch import float32
from read_image import image_read, image_write
from split_image import split_train_val_images


def get_args_parser():
    parser = argparse.ArgumentParser('Transpicker preprocessing', add_help=False)
    parser.add_argument('--split', default=False, type=bool, 
                    help='If need split the micrographs and the responding annotations. No more than 200 particles are recommended for each micrograph patch.')
    parser.add_argument('--is_equal_hist', default=True, type=bool,
                    help='If need do histogram equalization. Default is True.')
    parser.add_argument('--denoise_model', default='bi_filter', type=str, choices=('n2n','lowpass','gaussian','nlm','bi_filter'),
                    help='Choose a denoise model.')
    parser.add_argument('--root_dir', default='./data/empiar10028/', type=str,
                    help='Path to dataset.')
    parser.add_argument('--split_num', default=2, type=int,
                    help='The number of patches you want to split in each row and column.')
    parser.add_argument('--split_gap', default=200, type=int,
                    help='The overlap that needs to be left for segmentation. The recommended size of the interval is slightly larger than the particle diameter.')
    parser.add_argument('--iou_threshold', default=0.4, type=float,
                    help='bbox less than this threshold will not be saved.')
    parser.add_argument('--ext', default='.mrc', type=str,
                    help='The extention of micrograph file type.')

    return parser


def gaussian_filter(image, sigma=0, s=11):
    denoised = cv2.GaussianBlur(image, (s,s), sigma)
    return denoised


def lowpass_filter(image, factor=1):
    b, a = signal.butter(8, 0.2, 'lowpass')
    denoised = signal.filtfilt(b, a, image)
    return denoised


def bi_filter(image, d=0):
    denoised = cv2.bilateralFilter(image, d, 100, 5)
    return denoised


def topaz_denoise(image_path, output_path):
    # os.system('conda activate topaz')
    print(f'topaz denoise --patch-size 1024 -o {output_path} {image_path}')
    os.system(f'CUDA_VISIBLE_DEVICES=9 topaz denoise --patch-size 1024 -o {output_path} {image_path}')
    # os.system('conda activate transpicker')


def NLMeans_denoise(image):
    denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    return denoised


def equal_hist(image):
    gray = (image * 255).astype(np.uint8)
    gray = np.array(gray)
    # print(gray)
    equal_hist_image = cv2.equalizeHist(gray)

    return equal_hist_image


def preprocess(image_path, output_path, is_equal_hist=True, denoise_model=None):
    print('image_path: ', image_path)
    image = image_read(image_path)
    if not np.issubdtype(image.dtype, np.float32):
        image = image.astype(np.float32)
    image = (image - np.max(image))/(np.max(image) - np.min(image))
    # mean = np.mean(image)
    # sd = np.std(image)
    # image = (image - mean) / sd
    # image[image > 3] = 3
    # image[image < -3] = -3
    # equal hist to enhance the micrograph
    if is_equal_hist:
        image = equal_hist(image)
    # print(image.dtype)
    # denoise the micrograph
    if denoise_model == 'lowpass':
        image = lowpass_filter(image)
    elif denoise_model == 'n2n':
        image = topaz_denoise(image_path, output_path)
        return image
    elif denoise_model == 'gaussian':
        image = gaussian_filter(image)
    elif denoise_model == 'nlm':
        image = NLMeans_denoise(image)
    elif denoise_model == 'bi_filter':
        image = bi_filter(image)
    else:
        print('No available denoise model.')

    # save the preprocessed micrographs
    save_preprocessed_path = os.path.join(output_path, os.path.splitext(os.path.basename(image_path))[0]) + '_' + denoise_model + '.jpg'
    cv2.imwrite(save_preprocessed_path, image)

    return image



def preprocess_images(coco_dir, has_annots=True):
    pwd = os.getcwd()
    if has_annots:
        root_dir = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
        output_annot_dir = os.path.join(coco_dir, "preprocessed_annots")
        output_images_dir = os.path.join(coco_dir, "preprocessed_images")
        if not os.path.exists(output_annot_dir):
            os.makedirs(output_annot_dir)
        if not os.path.exists(output_images_dir):
            os.makedirs(output_images_dir)

        for sub_dir in os.listdir(coco_dir):
            if os.path.isdir(coco_dir + sub_dir) and not sub_dir.startswith("pre"):
                print(coco_dir + sub_dir)
                ann_dir = os.path.join(coco_dir + sub_dir, "annotations")
                print(f"cp {ann_dir}/* {output_annot_dir}")
                os.system(f"cp {ann_dir}/* {output_annot_dir}")
                img_dir = os.path.join(coco_dir + sub_dir, "micrographs")
                print("image_dir: ", img_dir)

                for img in os.listdir(img_dir):
                    image_path = os.path.join(img_dir, img)
                    if image_path.endswith("mrc"):
                        equal_hist_image = equal_hist(image_path, output_images_dir)

    else:
        root_dir = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
        coco_dir = os.path.join(root_dir, "coco/coco_split/test/")
        output_images_dir = os.path.join(coco_dir, "preprocessed_images")
        print(output_images_dir)
        if not os.path.exists(output_images_dir):
            os.makedirs(output_images_dir)

        for sub_dir in os.listdir(coco_dir):
            if sub_dir.endswith(".jpg"):
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

    img_files = []
    for root, directories, filenames in os.walk(img_dir, followlinks=True):
        for filename in filenames:
            if filename.endswith(("jpg", "png", "mrc", "mrcs", "tif", "tiff")) and not filename.startswith("."):
                img_files.append(os.path.join(root, filename))

    # Read annotations
    annotations = []
    for root, directories, filenames in os.walk(ann_dir, followlinks=True):
        for ann in sorted(filenames):
            if ann.endswith(("star", "box", "txt")) and not filename.startswith("."):
                annotations.append(os.path.join(root, ann))
    img_annot_pairs = find_image_annot_pairs(annotations, img_files)

    return img_annot_pairs


def main(args):
    root_dir = args.root_dir
    image_dir = root_dir + 'micrographs/'
    output_dir = root_dir + 'preprocessed/'

    # step 1: judge if split the micrographs and annotations
    if args.split:
        split_train_val_images(root_dir, root_dir+'split', args.split_num, args.split_gap, args.iou_threshold, args.ext)
        image_dir = root_dir + 'split/micrographs/'
        output_dir = root_dir + 'split/preprocessed/'

    # step 2: enhance and denoise the micrographs
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for image in os.listdir(image_dir):
        image_path = image_dir + image
        print(f'......Preprocess {image} now......')
        preprocess(image_path, output_dir, is_equal_hist=args.is_equal_hist, denoise_model=args.denoise_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Preprocessing script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)