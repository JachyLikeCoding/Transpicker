class global_val:
    box_size = 200
    denoise_method = 'n2n'
    cleaner_threshold = 0.6
    iou_threshold = 0.5
    
def set_box_size(box_size):
    global_val.box_size = box_size

def get_box_size():
    return global_val.box_size


def set_denoise_method(denoise_method):
    global_val.denoise_method = denoise_method

def get_denoise_method():
    return global_val.denoise_method


def set_cleaner_threshold(cleaner_threshold):
    global_val.cleaner_threshold = cleaner_threshold

def get_cleaner_threshold():
    return global_val.cleaner_threshold


def set_iou_threshold(iou_threshold):
    global_val.iou_threshold = iou_threshold

def get_iou_threshold():
    return global_val.iou_threshold