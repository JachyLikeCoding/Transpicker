import os
import sys
import cv2
import csv
import numpy as np
from PIL import Image
sys.path.append("/home/zhangchi/transpicker/Transpicker/src/transpicker")
sys.path.append(os.path.dirname(sys.path[0]))

from utils import BoundBox

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


def write_box(path, boxes, write_star=True):
    """
    Write box or star files.
    :param path: filepath or filename of the box file to write.
    :param boxes: boxes to write
    :param write_star: if true, a star file will be written.
    :return: None
    """
    if write_star:
        path = path[:-3] + 'star'
        write_star_file(path, boxes)
    else:
        write_eman_boxfile(path, boxes)


def write_star_file(path, boxes):
    with open(path, "w") as boxfile:
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
        boxwriter.writerow(["_rlnAutopickFigureOfMerit #5"])
        for box in boxes:
            boxwriter.writerow([box.x + box.w / 2, box.y + box.h / 2, -9999, -9999.00000, -9999.000000])


def write_eman_boxfile(path, boxes):
    with open(path, "w") as boxfile:
        boxwriter = csv.writer(
            boxfile, delimiter='\t', quotechar="|", quoting=csv.QUOTE_NONE
        )
        for box in boxes:
            # box.x, box,y = lower left corner
            boxwriter.writerow([box.x, box.y, box.w, box.h])


def write_cbox_file(path, boxes):
    with open(path, "w") as boxfile:
        boxwriter = csv.writer(
            boxfile, delimiter='\t', quotechar="|", quoting=csv.QUOTE_NONE
        )
        for box in boxes:
            est_w = box.meta["boxsize_estimated"][0]
            est_h = box.meta["boxsize_estimated"][1]
            boxwriter.writerow([box.x, box.y, box.w, box.h, est_w, est_h])


def get_star_file_header(file_name):
    """
    load the header information of star file.
    :param file_name:
    :return: list of head names, rows that are occupied by the header.
    """
    start_header = False
    header_names = []
    idx = None

    with open(file_name, "r") as read:
        for idx, line in enumerate(read.readlines()):
            if line.startswith("_"):
                if start_header:
                    header_names.append(line.strip().split()[0])
                else:
                    start_header = True
                    header_names.append(line.strip().split()[0])
            elif start_header:
                break
    if not start_header:
        raise IOError(f"No header information found in {file_name}")

    return header_names, idx


# read box file or star file or txt file
def read_eman_boxfile(path):
    """
    Read a box file in EMAN box format.
    :param path: the path of box file
    :return: List of bounding boxes.
    """
    boxes = []
    if os.path.getsize(path) == 0:
        print(path, " has no bbox.")
    else:
        boxreader = np.atleast_2d(np.genfromtxt(path))
        boxes = [BoundBox(x=box[0], y=box[1], w=box[2], h=box[3]) for box in boxreader]

    return boxes


def read_txt_file(path, box_width):
    boxreader = np.atleast_2d(np.genfromtxt(path))
    boxes = []
    for box in boxreader:
        bound_box = BoundBox(x=box[0] - box_width / 2, y=box[1] - box_width / 2, w=box_width, h=box_width)
        boxes.append(bound_box)
    return boxes


def read_star_file(path, box_width):
    header_names, skip_indices = get_star_file_header(path)
    boxreader = np.atleast_2d(np.genfromtxt(path, skip_header=skip_indices))
    boxes = []
    for box in boxreader:
        bound_box = BoundBox(x=box[0] - box_width / 2, y=box[1] - box_width / 2, w=box_width, h=box_width)
        boxes.append(bound_box)
    return boxes

# just for test how label percent affect the prediction results.
def read_percent_star_file(path, box_width, percent=100):
    from random import sample
    header_names, skip_indices = get_star_file_header(path)
    boxreader = np.atleast_2d(np.genfromtxt(path, skip_header=skip_indices))
    boxes = []
    for box in boxreader:
        bound_box = BoundBox(x=box[0] - box_width / 2, y=box[1] - box_width / 2, w=box_width, h=box_width)
        boxes.append(bound_box)
    box_num = int(len(boxes) * percent * 0.01)
    print(f'Before sample: {len(boxes)} boxes total.')
    boxes = sample(boxes,  box_num)
    print(f'After sample: {len(boxes)} boxes are chosen.')