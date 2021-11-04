import os
import numpy as np
from transpicker.coord_io import read_eman_boxfile, read_star_file
from transpicker.micrograph_cleaner import clean_micrograph, save_mask, display_compare


# delete boxes located in mask after micrograph cleaning.
def delete_bbox_in_mask(mask, boxes, scores, threshold=0.1): # TODO:测试几个阈值
    delete_indexes = []
    if_leave = []
    for i, box in enumerate(boxes):  # [box_xmin, box_ymin, box_xmax, box_ymax]
        box_center_x = int((box[0] + box[2]) / 2)
        box_center_y = int((box[1] + box[3]) / 2)
        if 0 < box_center_y < mask.shape[0] and 0 < box_center_x < mask.shape[1]:
            if mask[box_center_y][box_center_x] > threshold:  # 如果大于阈值，认为是碳膜或冰污染区域（非理想颗粒挑选区域）
                mask_i = mask[box_center_y][box_center_x]
                delete_indexes.append(i)
                if_leave.append(False)
            else:
                if_leave.append(True)
        else:  # 如果box的中心坐标超出mask边界，则直接删掉
            delete_indexes.append(i)
            if_leave.append(True)
    print('----------delete_box_indexes:', len(delete_indexes))
    boxes_cleaned = boxes[if_leave]
    saved_scores = scores[if_leave]
    return boxes_cleaned, saved_scores


def post_processing(boxes, scores, image_path, box_size=200, threshold=0.1):
    mask = clean_micrograph(image_path, box_size)
    save_mask(image_path, mask)
    # display_compare(image_path)
    saved_boxes, saved_scores = delete_bbox_in_mask(mask, boxes, scores, threshold)

    return saved_boxes, saved_scores
