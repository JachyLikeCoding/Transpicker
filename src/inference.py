# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import os
import cv2
import csv
import time
import json
import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torchvision.transforms as T
from transpicker.postprocess import post_processing
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

import util.misc as utils
import datasets.samplers as samplers
from util import box_ops
from models import build_model
from transpicker.utils import nms
from engine import evaluate, train_one_epoch
from datasets import build_dataset, get_coco_api_from_dataset
from transpicker.coord_io import write_box
from transpicker.read_image import image_read, image_write
torch.set_grad_enabled(False)

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


CLASSES = ['particle']

# colors for visualization
COLORS = ['red']


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu().numpy()
    b = b * np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    return b


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='cococryo')
    parser.add_argument('--coco_path', default='./data/empiar10028', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='./outputs/empiar10028_outputs/step_star_compare/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=27, type=int)
    parser.add_argument('--resume', default='./outputs/empiar10028_outputs/checkpoint0059.pth',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--box_size', default=200, type=int)

    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    parser.add_argument('--imgs_dir', default='./data/empiar10028/val/', type=str,
                        help='input images folder for inference')

    return parser


def main(args):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)
    print('device: ', device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # visual_model = False
    for img_file in os.listdir(args.imgs_dir):
        if os.path.isdir(args.imgs_dir + img_file):
            continue
        img_path = os.path.join(args.imgs_dir, img_file)
        out_imgname = args.output_dir + img_file[:-4] + '.png'

        if img_file.endswith('.mrc'):
            im = np.array(image_read(img_path), copy=False)
            im = Image.fromarray(im).convert('RGB')
        else:
            im = Image.open(img_path).convert('RGB')

        # mean-std normalize the input image (batch-size: 1)
        img = transform(im).unsqueeze(0)
        print('image shape: ', img.shape)
        img = img.to(device)

        start = time.time()
        start1 = time.time()
        # propagate through the model
        outputs = model(img)

        # from visualize_model import make_dot
        # if not visual_model:
        #     visual_model = True
        #     graph = make_dot(outputs, params=dict(list(model.named_parameters())))
        # # 第一个参数是模型的输出，第二个是模型的参数先列表化再字典化
        #     graph.view('model_structure.pdf', False, '/home/zhangchi/Deformable-DETR/')  # 第一个参数是文件名 第二个是保存路径

        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        probas = out_logits.sigmoid()
        print('probas:', probas)

        topk_values, topk_indexes = torch.topk(probas.view(out_logits.shape[0], -1), 300, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        keep = scores[0] > 0.2
        boxes = boxes[0, keep]
        labels = labels[0, keep]
        scores = scores[0, keep]

        # convert boxes from [0; 1] to image scales
        im_h, im_w = im.size
        target_sizes = torch.tensor([[im_w, im_h]])
        target_sizes = target_sizes.cuda()
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        end = time.time()
        print(f"[INFO] {end - start} time: {img_file} done!!!")

        # post-processing
        boxes = boxes.cpu().detach().numpy()
        boxes = boxes.squeeze(0)
        scores = scores.cpu().detach().numpy()
        save_boxes(boxes, scores, img_file, im_h, "_transpicker")
        boxes, scores = nms(boxes, scores, threshold=0.6)
        save_boxes(boxes, scores, img_file, im_h, "_transpicker_nms")
        boxes, scores = post_processing(boxes, scores, img_path, args.box_size, threshold=0.1)

        boxes_scores = []
        for i, box in enumerate(boxes):
            boxes_scores.append([box[0], box[1], box[2], box[3], scores[i]])

        # save box files
        write_name = args.output_dir+img_file[:-4]+'.star'
        # write_box(write_name, boxes, write_star=True)
        with open(write_name, "w") as boxfile:
            boxwriter = csv.writer(
                boxfile, delimiter='\t', quotechar="|", quoting=csv.QUOTE_NONE
            )
            boxwriter.writerow([])
            boxwriter.writerow(["data_"])
            boxwriter.writerow([])
            boxwriter.writerow(["loop_"])
            boxwriter.writerow(["_rlnCoordinateX #1 "])
            boxwriter.writerow(["_rlnCoordinateY #2 "])
            boxwriter.writerow(["_rlnClassNumber #3 "])
            boxwriter.writerow(["_rlnAnglePsi #4"])
            boxwriter.writerow(["_rlnScore #5"])
            for box in boxes_scores:
                boxwriter.writerow([(box[0] + box[2]) / 2, im_h - (box[1] + box[3]) / 2, -9999, -9999.00000, box[4]])

        if img_file.endswith('.mrc'):
            im = np.array(image_read(img_path), copy=False)
            source_img = Image.fromarray(im).convert('RGB')
            image = source_img
            mean = np.mean(image)
            sd = np.std(image)
            image = (image - mean) / sd
            image[image > 3] = 3
            image[image < -3] = -3

        else:
            source_img = Image.open(img_path).convert("RGB")
            source_img = source_img.transpose(Image.FLIP_TOP_BOTTOM)

        draw = ImageDraw.Draw(source_img)
        i = 0

        for xmin, ymin, xmax, ymax in boxes.tolist():
            x = int((xmax + xmin) / 2)
            y = int((ymin + ymax) / 2)
            draw.rectangle(((xmin, im_h - ymin), (xmax, im_h - ymax)), outline=(255, 255, 0), width=3)
            # radius = int((xmax - xmin) / 2)

            # c = Circle((x, y), radius, fill=False, color='r')
            # ax.add_patch(c)
            # print('--------')
            print('i= ', i)
            # print('label is = ', label_list[i]-1)
            i += 1
        end1 = time.time()
        print(f"----[INFO] {end1 - start1} time: with cleaning {img_file} done!!!")
        source_img.save(out_imgname, "png")
        # plt.savefig(out_imgname)

    results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
    print("Outputs", results)



def save_boxes(boxes, scores, img_file, im_h, out_imgname):
    # save box files
    write_name = args.output_dir + img_file[:-4] + out_imgname + '.star'
    # write_box(write_name, boxes, write_star=True)
    with open(write_name, "w") as boxfile:
        boxwriter = csv.writer(
            boxfile, delimiter='\t', quotechar="|", quoting=csv.QUOTE_NONE
        )
        boxwriter.writerow([])
        boxwriter.writerow(["data_"])
        boxwriter.writerow([])
        boxwriter.writerow(["loop_"])
        boxwriter.writerow(["_rlnCoordinateX #1 "])
        boxwriter.writerow(["_rlnCoordinateY #2 "])
        boxwriter.writerow(["_rlnClassNumber #3 "])
        boxwriter.writerow(["_rlnAnglePsi #4"])
        boxwriter.writerow(["_rlnScore #5"])
        for i, box in enumerate(boxes):
            boxwriter.writerow([(box[0] + box[2]) / 2, im_h - (box[1] + box[3]) / 2, -9999, -9999.00000, scores[i]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
