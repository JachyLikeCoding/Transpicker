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

Then you can run `python make_coco_dataset.py` to get coco-style datasets as following:
```
code_root/
└── data/
    └── empiarxxxxx/
        ├── micrographs/
            ├── 0001.mrc
            ├── 0002.mrc
        	└── xxxx.mrc
        └── coordinates/
        	├── 0001.star
            ├── 0002.star
        	└── xxxx.star
        └── annotations/
            ├── instances_train.json
            └── instances_val.json
```


### Training
The training step can be run by 
> python train.py

### Predict
The particle prediction can be run by 
> python predict.py

## GUI usage
If you want to use GUI to display the micrographs and choose better thresholds, you can use 
> python boxmanager.py

to activate the GUI panel.