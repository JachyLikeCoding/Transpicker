# Transpicker
By Chi Zhang, Hongjia Li, Xiaohua Wan, Xuemei Chen, Jieqing Feng, Fa Zhang.
This repository is an official implementation of the paper:
Transpicker: a transformer-based particle picking framework for cryoEM images.

## Introduction
A 2D particle picker for cryoEM micrographs.
### Abstract

## License

## Citing TransPicker
If you find TransPicker useful in your research, please consider citing:

## Installation
### Requirements
- Linux, CUDA>=9.2
- Python>=3.7
In order to run the TransPicker we need to install all required packages.
This can be done by creating a virtual environment with `python3 -m venv env` and activating it with `source ./env/bin/activate`. Once the virtual Python environment is activated, the required packages can be installed with pip using `pip install -r requirements.txt`.


## Usage
To get a complete description of usage execute
> transpicker -h

### Dataset preparation
You can download datasets from EMPIAR or using your own dataset and organize them as following:
```
code_root/
└── data/
    └── empiarxxxxx/
        ├── micrographs/
            ├── 0001.mrc
            ├── 0002.mrc
        	└── xxxx.mrc
        └── annots/
        	├── 0001.star
            ├── 0002.star
        	└── xxxx.star
```

You can name your own dataset in other ways, but the `micrographs` and `annots` sub directory should be metained (or change the source code).
### Make coco-style dataset for training and testing
Then you can run `python make_coco_dataset.py` to get coco-style datasets as following:
```
code_root/
└── data/
    └── empiarxxxxx/
        ├── micrographs/
            ├── 0001.mrc
            ├── 0002.mrc
        	└── xxxx.mrc
        └── annots/
        	├── 0001.star(or .box or .txt)
            ├── 0002.star
        	└── xxxx.star
        └── annotations/
            ├── instances_train.json
            └── instances_val.json
```

### Preprocessing
Before training a particle-picking model, you'd better preprocess your datasets using the `preprocess.py` script.
The preprocess step can be run by:
> python preprocess.py

Available optionals:
```
usage: Preprocessing script [-h] [--split SPLIT]
                            [--is_equal_hist IS_EQUAL_HIST]
                            [--denoise_model {n2n,lowpass,gaussian,nlm,bi_filter}]
                            [--root_dir ROOT_DIR] [--split_num SPLIT_NUM]
                            [--split_gap SPLIT_GAP]
                            [--iou_threshold IOU_THRESHOLD] [--ext EXT]

optional arguments:
  -h, --help            show this help message and exit
  --split SPLIT         If need split the micrographs and the responding
                        annotations. No more than 200 particles are
                        recommended for each micrograph patch.
  --is_equal_hist IS_EQUAL_HIST
                        If need do histogram equalization. Default is True.
  --denoise_model {n2n,lowpass,gaussian,nlm,bi_filter}
                        Choose a denoise model.
  --root_dir ROOT_DIR   Path to dataset.
  --split_num SPLIT_NUM
                        The number of patches you want to split in each row
                        and column.
  --split_gap SPLIT_GAP
                        The overlap that needs to be left for segmentation.
                        The recommended size of the interval is slightly
                        larger than the particle diameter.
  --iou_threshold IOU_THRESHOLD
                        bbox less than this threshold will not be saved.
  --ext EXT             The extention of micrograph file type.
```

### Training
The training step can be run by:
> cd bin \
> sh train.sh

You can use `-h` to view all parameters.
### Prediction
The particle prediction can be run by 
> python predict.py

### GUI usage
If you want to use GUI to display the micrographs and choose better thresholds, you can use:
> python boxmanager.py

to activate the GUI panel.