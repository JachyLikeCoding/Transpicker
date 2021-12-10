import argparse
import os
import json
import numpy as np
import multiprocessing
from gooey import Gooey, GooeyParser


def create_parser(parser):
    required_groups = parser.add_argument_group(
        "Train arguments","These options are mandatory to run transpicker train."
    )

    required_groups.add_argument(
        "--image_files",
        required=True,
        widget="FileChooser",
        help="Training image files."
    )

    required_groups.add_argument(
        "--annotations",
        required=True,
        help="Path to folder containing your annotation json files.",
        default="",
        gooey_options={
            "validator": {
                "test": 'user_input.endswith("json")',
                "message": "File has to end with .json",
            },
            "default_file": "annotations.json",
        },
        widget="DirChooser",
    )

    required_groups.add_argument(
        "--outputs",
        required=True,
        widget="FileChooser",
        help="Annotation files."
    )

    required_groups.add_argument(
        "--saved_model_name",
        default="",
        help="path for saving final weights of model.",
        widget="FileSaver",
        # gooey_options={
        #     "validator": {
        #         "test": 'user_input.endswith("h5")',
        #         "message": "File has to end with .h5",
        #     },
        #     "default_file": "transpicker_model.h5",
        # }
    )


    required_groups.add_argument("--epochs",type=int,required=True,default=70,help="Number of training epochs.",)
    required_groups.add_argument('--lr', default=2e-4, type=float)
    required_groups.add_argument('--lr_backbone', default=2e-5, type=float)
    required_groups.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    required_groups.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    required_groups.add_argument('--batch_size', default=2, type=int)
    required_groups.add_argument('--weight_decay', default=1e-4, type=float)
    required_groups.add_argument('--lr_drop', default=40, type=int)
    required_groups.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    required_groups.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')


    ################################################################################## model group
    config_model_group = parser.add_argument_group(
        "Model options",
        "Options to configure your model. The default options are well tested and typically don't have to be changed.",
    )

    config_model_group.add_argument(
        "-a", "--architecture",
        default="deformable-detr",
        choices=["detr", "deformable-detr"],
        help="Backend network architecture.",
    )

    config_model_group.add_argument(
        "-i", "--input_size",
        default=2048,
        nargs="+",
        help="The shorter image dimension is downsized to this size and the long size according the aspect ratio."
             "This is not the size of your micrographs and only rarely needs to be changes."
             "You can also specify height and width of the input image size by seperating them by a whitespace.",
    )

    # * Backbone
    config_backbone_group = parser.add_argument_group(
        "Backbone options",
        "Options to configure your model. The default options are well tested and typically don't have to be changed.",
    )
    config_backbone_group.add_argument('--backbone', default='resnet50', type=str, help="Name of the convolutional backbone to use")
    config_backbone_group.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    config_backbone_group.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    config_backbone_group.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    config_backbone_group.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")

    # * Transformer
    config_transformer_group = parser.add_argument_group(
        "Transformer options",
        "Options to configure your model. The default options are well tested and typically don't have to be changed.",
    )
    config_transformer_group.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    config_transformer_group.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    config_transformer_group.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    config_transformer_group.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    config_transformer_group.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    config_transformer_group.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    config_transformer_group.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")

    config_transformer_group.add_argument('--dec_n_points', default=4, type=int)
    config_transformer_group.add_argument('--enc_n_points', default=4, type=int)

    # Loss
    config_loss_group = parser.add_argument_group(
        "Loss options",
        "Options to configure your model. The default options are well tested and typically don't have to be changed.",
    )
    config_loss_group.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    config_loss_group.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    config_loss_group.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    config_loss_group.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    config_loss_group.add_argument('--mask_loss_coef', default=1, type=float)
    config_loss_group.add_argument('--dice_loss_coef', default=1, type=float)
    config_loss_group.add_argument('--cls_loss_coef', default=2, type=float)
    config_loss_group.add_argument('--bbox_loss_coef', default=5, type=float)
    config_loss_group.add_argument('--giou_loss_coef', default=2, type=float)
    config_loss_group.add_argument('--focal_alpha', default=0.25, type=float)


def get_parser():
    '''
    Create parser
    :return: parser
    '''
    parser = GooeyParser(
        description = "Train transpicker model.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    create_parser(parser)
    return parser