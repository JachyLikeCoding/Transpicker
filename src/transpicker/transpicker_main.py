import argparse
import os
import __init__ as ini
import wx
import boxmanager
from gooey import Gooey, GooeyParser
# import config_tools


def config_configuration_parser(parser_config):
    ######################################################################### required group
    config_required_group = parser_config.add_argument_group(
        "Datasets arguments",
        "Make coco-format datasets for transpicker training.",
    )
    config_required_group.add_argument(
        '--dataset_path',
        widget="DirChooser",
        help="Path to the dataset you want to deal with. (e.g. empair10028)",
        default='',
        type=str,
    )
    config_required_group.add_argument(
        '--images_path',
        widget="DirChooser",
        default='',
        help="Path to the image folder containing the images to train on.",
        type=str,
    )
    config_required_group.add_argument(
        '--phase',
        default='train',
        choices=["train", "val"],
        type=str,
    )
    config_required_group.add_argument(
        '--input_size',
        default=1024,
        nargs="+",
        help="This is not the size of your micrographs. The shorter dimension of image is downsized to this size.",
        type=int,
    )
    config_required_group.add_argument(
        "--particle_size",
        type=int,
        default=200,
        help="You should specify the same box size here as you used in your training data. "
             "If train on several datasets, use the average boxsize.",
    )

    ########################################################################## denoised group
    config_denoising_group = parser_config.add_argument_group(
        "Denoising options",
        "By default crYOLO denoises your data before training/picking. It will use a low-pass filter, but you can also use neural network denoising (JANNI). Default options are good and typically don't have to be changed.",
    )

    config_denoising_group.add_argument(
        "--filtered_output",
        default="filtered_tmp/",
        help="Output folder for filtered images",
        widget="DirChooser",
    )

    config_denoising_group.add_argument(
        "-f",
        "--filter",
        default="LOWPASS",
        help="Noise filter applied before training/picking. You can choose between a normal low-pass filter"
             " and neural network denoising (JANNI).",
        choices=["NONE", "LOWPASS", "JANNI"],
    )

    config_denoising_group.add_argument(
        "--low_pass_cutoff",
        type=float,
        default=0.1,
        help="Low pass filter cutoff frequency",
        gooey_options={
            "validator": {
                "test": "0.0 <= float(user_input) <= 0.5",
                "message": "Must be between 0 and 0.5",
            }
        },
    )
    config_denoising_group.add_argument(
        "--janni_model",
        help="Path to JANNI model",
        widget="FileChooser",
        gooey_options={"wildcard": "*.h5"},
        default=None,
    )

    config_denoising_group.add_argument(
        "--janni_overlap",
        type=int,
        default=24,
        help="Overlap of patches in pixels (only needed when using JANNI)",
    )

    config_denoising_group.add_argument(
        "--janni_batches",
        type=int,
        default=3,
        help="Number of batches (only needed when using JANNI)",
    )


def config_boxmanager_parser(parser_config):
    config_display_group = parser_config.add_argument_group(
        "Display options",
        "Options to display your micrographs and re-pick particles.",
    )

    config_display_group.add_argument(
        "-i", "--image_dir", help="Path to image directory.", widget="DirChooser"
    )
    config_display_group.add_argument(
        "-b", "--box_dir", help="Path to box directory.", widget="DirChooser"
    )
    config_display_group.add_argument(
        "-s", "--box_size", type=int, default=200,
        help="Box size to display. Please set according to your datasets.",
    )
    config_display_group.add_argument(
        "-pt", "--prob_threshold", default=0.8,
        help="The probability threshold can decide how many bounding boxes will be displayed and saved. ",
    )


def create_parser():
    parent_parser = GooeyParser(
        description="The transpicker particle picking procedure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parent_parser.add_subparsers(help="sub-command help")

    # Config generator
    parser_config = subparsers.add_parser(
        "config", help="train help",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    config_configuration_parser(parser_config)

    # boxmanager
    parser_display = subparsers.add_parser(
        "display", help="display manager help",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    config_boxmanager_parser(parser_display)

    # Training parser
    parser_train = subparsers.add_parser(
        "train",
        help="train help",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    from train import create_parser as create_train_parser
    create_train_parser(parser_train)


    # Picking parser
    parser_pick = subparsers.add_parser(
        "predict",
        help="predict help",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    from predict import create_parser as create_predict_parser
    create_predict_parser(parser_pick)


    # Evaluation parser
    parser_eval = subparsers.add_parser(
        "evaluation",
        help="evaluation help",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )


    return parent_parser


PARSER = None


def main():
    import sys

    args = PARSER.parse_args()

    print("########################################################################################")
    print("Important debugging information.\nIn case of any problem, please provide this information.")
    print("########################################################################################")
    for cmd_arg in sys.argv:
        if cmd_arg.startswith("-"):
            print("")
        print(cmd_arg, end=" ")
    print("")
    print("########################################################################################")

    if "display" in sys.argv[1]:
        boxmanager.start_boxmanager(args.image_dir, args.box_dir, args.box_size, args.prob_threshold)

    elif "config" in sys.argv[1]:
        import utils
        filter = None
        if args.filter == "LOWPASS":
            filter = [args.low_pass_cutoff, args.filtered_output]
        elif args.filter == "JANNI":
            if args.janni_model is None:
                print("Please specify the JANNI model file.")
                sys.exit(1)
            filter = [
                args.janni_model,
                args.janni_overlap,
                args.janni_batches,
                args.filtered_output,
            ]
        if type(args.input_size) == list:
            if len(args.input_size) == 2:
                args.input_size = [int(args.input_size[0]), int(args.input_size[1])]
            elif len(args.input_size) == 1:
                args.input_size = int(args.input_size[0])

        # config_tools.generate_config_file(
        #     config_out_path=args.config_out_path,
        #     input_size=args.input_size,
        #     max_box_per_image=700,
        #     filter=args.filter,
        #     train_annot_folder=args.train_annot_folder,
        #     train_image_folder=args.train_image_folder,
        #     batch_size=args.batch_size,
        #     learning_rate=args.learing_rate,
        #     log_path=args.log_path,
        #     valid_times=1,
        #     valid_image_folder=args.valid_image_folder,
        #     valid_annot_folder=args.valid_annot_folder,
        #     normalization=args.norm
        # )

    elif "train" in sys.argv[1]:
        import train
        train.main(args)

    elif "predict" in sys.argv[1]:
        import predict
        predict.main(args)

    elif "evaluation" in sys.argv[1]:
        import eval


def _main_():
    global PARSER
    import sys

    PARSER = create_parser()

    if len(sys.argv) >= 2:
        if not "--ignore-gooey" in sys.argv:
            sys.argv.append("--ignore-gooey")

    kwargs = {"terminal_font_family": "monospace", "richtext_controls": True}

    Gooey(
        main,
        program_name="TransPicker" + ini.__version__,
        image_dir=os.path.join(os.path.abspath(os.path.dirname(__file__)), "../icons"),
        progress_regex=r"^.* \( Progress:\s+(-?\d+) % \)$",
        disable_progress_bar_animation=True,
        tabbed_groups=True,
        default_size=(1024, 680),
        **kwargs
    )()


if __name__ == "__main__":
    _main_()
