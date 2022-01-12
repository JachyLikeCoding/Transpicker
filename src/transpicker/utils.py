import sys
import json
import numpy as np
import urllib.request
import multiprocessing
from enum import Enum
from sklearn import mixture

class BoundBox:
    """
    Bounding box of a particle.
    """

    def __init__(self, x, y, w, h, c=None, classes=None):
        """
        creates a bounding box.
        :param x: x coordinate of particle center.
        :param y: y coordinate of particle center.
        :param w: width of box
        :param h: height of box
        :param c: confidence of the box
        :param classes: class of the bounding box object
        """
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.c = c
        self.classes = classes
        self.meta = {}
        self.label = -1
        self.score = -1
        self.info = None

    def get_label(self):
        """

        :return: class with highest probability
        """
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        """

        :return: probability of the class
        """
        self.score = self.classes[self.get_label()]

        return self.score


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def normalize(image, margin_size=0):
    """
    Normalize an image.
    :param image: image to normalize
    :param margin_size: relative margin size to be ignored during normalization. Number between 0-1.
    :return: normalized image
    """
    if margin_size < 0 or margin_size > 1:
        print("Normalization has to be between 0 and 1.")
        if margin_size < 0:
            margin_size = 0
        else:
            margin_size = 1
        print("Has set it to", margin_size)

    if not np.issubdtype(image.dtype, np.float32):
        image = image.astype(np.float32)

    mask = np.s_[
           int(image.shape[0] * margin_size):int(image.shape[0] * (1 - margin_size)),
           int(image.shape[1] * margin_size):int(image.shape[1] * (1 - margin_size)),
           ]
    img_mean = np.mean(image[mask])
    img_std = np.std(image[mask])

    image = (image - img_mean) / (3 * 2 * img_std + 0.000001)

    return image


def normalize_gmm(image, margin_size=0):
    """
    Normalize an image with gaussian mixture model.
    :param image: image to normalize
    :param margin_size: relative margin size to be ignored during normalization. Number between 0-1.
    :return: normalized image
    """
    if margin_size < 0 or margin_size > 1:
        print("Normalization has to be between 0 and 1.")
        if margin_size < 0:
            margin_size = 0
        else:
            margin_size = 1
        print("Has set it to", margin_size)

    if not np.issubdtype(image.dtype, np.float32):
        image = image.astype(np.float32)

    mask = np.s_[
           int(image.shape[0] * margin_size):int(image.shape[0] * (1 - margin_size)),
           int(image.shape[1] * margin_size):int(image.shape[1] * (1 - margin_size)),
           ]

    clf = mixture.GaussianMixture(n_components=2, covariance_type='diag')
    clf.fit(np.expand_dims(image[mask].ravel()[::15], axis=1))

    if clf.means_[1, 0] > clf.means_[0, 0]:
        mean = clf.means_[1, 0]
        var = clf.covariances_[1, 0]
    else:
        mean = clf.means_[0, 0]
        var = clf.covariances_[0, 0]

    image = (image - mean) / (3 * 2 * np.sqrt(var) + 0.000001)

    return image


def bbox_iou(box1, box2):
    x1_min = box1.x - box1.w / 2
    x1_max = box1.x + box1.w / 2
    y1_min = box1.y - box1.h / 2
    y1_max = box1.y + box1.h / 2

    x2_min = box2.x - box2.w / 2
    x2_max = box2.x + box2.w / 2
    y2_min = box2.y - box2.h / 2
    y2_max = box2.y + box2.h / 2

    intersect_w = interval_overlap([x1_min, x1_max], [x2_min, x2_max])
    intersect_h = interval_overlap([y1_min, y1_max], [y2_min, y2_max])

    intersect = intersect_w * intersect_h

    union = box1.w * box1.h + box2.w * box2.h - intersect
    iou = intersect / union


def interval_overlap(interval_a, interval_b):
    """
    calculate the overlap between two intervals.
    :param interval_a: tuple with two elements (lower and upper bound)
    :param interval_b: tuple with two elements (lower and upper bound)
    :return: overlap between two intervals
    """
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def nms(bounding_boxes, confidence_scores, threshold=0.6):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_scores)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_scores[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    picked_boxes = np.array(picked_boxes).squeeze()
    picked_score = np.array(picked_score)

    return picked_boxes, picked_score


def filter_images_noise2noise_dir(
        imgs_path,
        output_dir_filtered_imgs,
        model_path,
        padding=15,
        batch_size=4,
        resize_to=None,
):
    import h5py
    import janni
    from janni import predict as janni_predict
    from janni import models as janni_models
    # TODO:JANNI DENOISE

    with h5py.File(model_path, mode='r') as f:
        try:
            import numpy as np
            model = str(np.array((f["model_name"])))
            patch_size = tuple(f["patch_size"])
        except KeyError:
            print("Not supported filtering model.")
            sys.exit(0)

    if model == "unet":
        model = janni_models.get_model_unet(input_size=patch_size)
        model.load_weights(model_path)
    else:
        print("Not supported model ", model)
        sys.exit(0)

    filtered_paths = janni_predict.predict_list(
        image_paths=imgs_path,
        output_path=output_dir_filtered_imgs,
        model=model,
        patch_size=patch_size,
        padding=padding,
        batch_size=batch_size,
        output_resize_to=resize_to,
    )

    return filtered_paths


def write_command(filename, commands):
    try:
        import os
        os.makedirs(os.path.dirname(filename))
    except Exception:
        pass

    text_file = open(filename, "w")
    text_file.write(commands)
    text_file.close()