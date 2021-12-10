import argparse
import os
import sys
import csv
import time
import json
import torch
import random
import numpy as np
import multiprocessing
import torchvision.transforms as T
import util.misc as utils
from transpicker.postprocess import post_processing
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from gooey import Gooey, GooeyParser
from util import box_ops
from models import build_model
from transpicker.utils import nms
# import denoise
# import read_image
# import config_tools

torch.set_grad_enabled(False)

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def create_parser(parser):
    required_groups = parser.add_argument_group(
        "Required arguments", "These options are mandatory to run transpicker train."
    )
    # dataset parameters
    required_groups.add_argument('--dataset_file', default='cococryo',
                                 required=True, widget="FileChooser", )
    required_groups.add_argument('--coco_path', default='/home/zhangchi/cryodata/empiar10590', type=str,
                                 required=True, widget="FileChooser", )
    required_groups.add_argument('--imgs_dir', default='/home/zhangchi/cryodata/empiar10590/val/', type=str,
                                 help='input images folder for inference')
    required_groups.add_argument('--output_dir',
                                 default='./transpicker_outputs/my_outputs_10590_100_denoised/pre_nms_mask/',
                                 required=True, widget="FileChooser",
                                 help='path where to save, empty for no saving')
    required_groups.add_argument('--device', default='cuda',
                                 help='device to use for training / testing')
    required_groups.add_argument('--seed', default=42, type=int)
    required_groups.add_argument('--resume',
                                 default='./transpicker_outputs/my_outputs_10590_100_denoised/checkpoint0069.pth',
                                 widget="FileChooser",
                                 help='resume from checkpoint')
    required_groups.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    required_groups.add_argument('--eval', action='store_true')
    required_groups.add_argument('--box_size', default=200, type=int)
    required_groups.add_argument('--num_workers', default=2, type=int)
    required_groups.add_argument('--cache_mode', default=False, action='store_true',
                                 help='whether to cache images on memory')

    ######################################################################################## preprocess group
    config_preprocess_group = parser.add_argument_group(
        "Preprocess options",
        "Options to pre-process your micrographs.",
    )

    config_preprocess_group.add_argument(
        "-d", "--denoise",
        default="lowpass",
        choices=["none", "n2n", "lowpass", "n2v"],
        help="Choose denoise methods.",
    )

    config_preprocess_group.add_argument(
        "--lowpass_filter",
        type=float,
        default=0.1,
        help="Low pass filter cutoff frequency.",
        gooey_options={
            "validator": {
                "test": "0.0<=float(user_input)<=0.5",
                "message": "Must be between 0 and 0.5.",
            }
        },
    )

    config_preprocess_group.add_argument(
        "--n2n_model",
        help="Path to n2n model.",
        widget="FileChooser",
        gooey_options={"wildcard": "*.h5"},
        default=None,
    )

    config_preprocess_group.add_argument(
        "--equal_hist",
        type=bool,
        default=True,
        choices=[True, False],
        help="Choose if need equal hist or not.",
    )

    config_preprocess_group.add_argument(
        "--preprocessed_output",
        default="preprocessed_tmp/",
        help="Output folder for preprocessed images",
        widget="DirChooser",
    )

    ######################################################################################## postprocess group
    config_postprocess_group = parser.add_argument_group(
        "Post-process options",
        "Options to post-process your micrographs.",
    )

    config_postprocess_group.add_argument(
        "--cleaner",
        type=bool,
        default=True,
        choices=[True, False],
        help="If need micrograph-cleaner to postprocess your images.",
    )

    config_postprocess_group.add_argument(
        "--mask_threshold",
        default=0.1,
        nargs="+",
        help="Mask threshold used for micrograph-cleaner.If > threshold, be regarded as undesirable regions for "
             "particle picking.",
    )

    config_postprocess_group.add_argument(
        "--cleaner_boxsize",
        default=200,
        nargs="+",
        help="Box size used for micrograph-cleaner.",
    )

    config_postprocess_group.add_argument(
        "--mask_output",
        default="mask_tmp/",
        widget="DirChooser",
        help="Mask save output path used for micrograph-cleaner.",
    )
    #################################################################### model arguments
    config_model_groups = parser.add_argument_group(
        "Model arguments", "Just maintain the same setting as training."
    )
    config_model_groups.add_argument('--lr', default=2e-4, type=float)
    config_model_groups.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    config_model_groups.add_argument('--lr_backbone', default=2e-5, type=float)
    config_model_groups.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'],
                                     type=str, nargs='+')
    config_model_groups.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    config_model_groups.add_argument('--batch_size', default=2, type=int)
    config_model_groups.add_argument('--weight_decay', default=1e-4, type=float)
    config_model_groups.add_argument('--epochs', default=50, type=int)
    config_model_groups.add_argument('--lr_drop', default=40, type=int)
    config_model_groups.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    config_model_groups.add_argument('--clip_max_norm', default=0.1, type=float,
                                     help='gradient clipping max norm')

    config_model_groups.add_argument('--sgd', action='store_true')
    # Variants of Deformable DETR
    config_model_groups.add_argument('--with_box_refine', default=False, action='store_true')
    config_model_groups.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    config_model_groups.add_argument('--frozen_weights', type=str, default=None,
                                     help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    config_model_groups.add_argument('--backbone', default='resnet50', type=str,
                                     help="Name of the convolutional backbone to use")
    config_model_groups.add_argument('--dilation', action='store_true',
                                     help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    config_model_groups.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                                     help="Type of positional embedding to use on top of the image features")
    config_model_groups.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                                     help="position / size * scale")
    config_model_groups.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    # * Segmentation
    config_model_groups.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # * Transformer
    config_model_groups.add_argument('--enc_layers', default=6, type=int,
                                     help="Number of encoding layers in the transformer")
    config_model_groups.add_argument('--dec_layers', default=6, type=int,
                                     help="Number of decoding layers in the transformer")
    config_model_groups.add_argument('--dim_feedforward', default=1024, type=int,
                                     help="Intermediate size of the feedforward layers in the transformer blocks")
    config_model_groups.add_argument('--hidden_dim', default=256, type=int,
                                     help="Size of the embeddings (dimension of the transformer)")
    config_model_groups.add_argument('--dropout', default=0.1, type=float,
                                     help="Dropout applied in the transformer")
    config_model_groups.add_argument('--nheads', default=8, type=int,
                                     help="Number of attention heads inside the transformer's attentions")
    config_model_groups.add_argument('--num_queries', default=300, type=int,
                                     help="Number of query slots")
    config_model_groups.add_argument('--dec_n_points', default=4, type=int)
    config_model_groups.add_argument('--enc_n_points', default=4, type=int)

    # Loss
    config_model_groups.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                                     help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    config_model_groups.add_argument('--set_cost_class', default=2, type=float,
                                     help="Class coefficient in the matching cost")
    config_model_groups.add_argument('--set_cost_bbox', default=5, type=float,
                                     help="L1 box coefficient in the matching cost")
    config_model_groups.add_argument('--set_cost_giou', default=2, type=float,
                                     help="giou box coefficient in the matching cost")

    # * Loss coefficients
    config_model_groups.add_argument('--mask_loss_coef', default=1, type=float)
    config_model_groups.add_argument('--dice_loss_coef', default=1, type=float)
    config_model_groups.add_argument('--cls_loss_coef', default=2, type=float)
    config_model_groups.add_argument('--bbox_loss_coef', default=5, type=float)
    config_model_groups.add_argument('--giou_loss_coef', default=2, type=float)
    config_model_groups.add_argument('--focal_alpha', default=0.25, type=float)


def get_parser():
    '''
    Create parser
    :return: parser
    '''
    parser = GooeyParser(
        description="Picking particles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    create_parser(parser)
    return parser


def _main_():
    if len(sys.argv) >= 2:
        if not "--ignore-gooey" in sys.argv:
            sys.argv.append("--ignore-gooey")

    kwargs = {"terminal_font_family": "monospace", "richtext_controls": True}
    Gooey(
        main,
        program_name='Transpicker Predict',
        image_dir=os.path.join(os.path.abspath(os.path.dirname(__file__)), "../icons"),
        progress_regex=r"^.* \( Progress:\s+(-?\d+) % \)$",
        disable_progress_bar_animation=True,
        tabbed_groups=True,
        **kwargs
    )()


def main(args=None):
    if args is None:
        parser = get_parser()
        args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    inference(args)


def inference(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    # if args.frozen_weights is not None:
    #     assert args.masks, "Frozen training is meant for segmentation only"

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

    for img_file in os.listdir(args.imgs_dir):
        if os.path.isdir(args.imgs_dir + img_file):
            continue
        img_path = os.path.join(args.imgs_dir, img_file)
        out_imgname = args.output_dir + img_file[:-4] + '.png'
        im = Image.open(img_path).convert('RGB')
        # im = np.array(image_read(img_path), copy=False)
        # mean-std normalize the input image (batch-size: 1)
        img = transform(im).unsqueeze(0)
        print('image shape: ', img.shape)
        img = img.to(device)

        start = time.time()
        # propagate through the model
        outputs = model(img)

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
        boxes, scores = nms(boxes, scores, threshold=0.1)
        boxes, scores = post_processing(boxes, scores, img_path, args.box_size, threshold=0.1)

        boxes_scores = []
        for i, box in enumerate(boxes):
            boxes_scores.append([box[0], box[1], box[2], box[3], scores[i]])

        # save box files
        write_name = args.output_dir + img_file[:-4] + '.star'
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

        # plot_results
        # _,ax = plt.subplots(figsize=(16,16))
        source_img = Image.open(img_path).convert("RGB")
        # source_img = np.array(image_read(img_path), copy=False)
        source_img = source_img.transpose(Image.FLIP_TOP_BOTTOM)

        draw = ImageDraw.Draw(source_img)
        # ax.imshow(source_img, cmap='Greys_r', vmin=-3.5, vmax=3.5, interpolation='bilinear')
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
        source_img.save(out_imgname, "png")
        # plt.savefig(out_imgname)

    results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
    print("Outputs", results)


if __name__ == '__main__':
    _main_()
