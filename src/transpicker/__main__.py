'''TransPicker particle picking'''

from . import main

def main():
    import argparse, os
    parser = argparse.ArgumentParser(description=__doc__)
    import transpicker
    parser.add_argument('--version', action="version", version='TransPicker'+transpicker.__version__)

    import transpicker.make_coco_dataset
    import transpicker.postprocess
    import transpicker.predict
    import transpicker.read_image
    import transpicker.preprocess
    import transpicker.split_image
    import transpicker.train
    import transpicker.stitch_image
    import transpicker.transpicker_main

    modules = [transpicker.make_coco_dataset,
            transpicker.predict,
            transpicker.predict,
            transpicker.read_image,
            transpicker.preprocess,
            transpicker.split_image,
            transpicker.train,
            transpicker.stitch_image,
            transpicker.transpicker_main,
            ]

    subparsers = parser.add_subparsers(title='Choose a command')
    subparsers.required = 'True'

    def get_str_name(module):
        return os.path.splitext(os.path.basename(module.__file__))[0]

    for module in modules:
        this_parser = subparsers.add_parser(get_str_name(module), description=module.__doc__)
        module.add_args(this_parser)
        this_parser.set_defaults(func=module.main)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main(prog_name="transpicker")