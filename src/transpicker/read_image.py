import os
import sys
import cv2
import imageio
import numpy as np
import mrcfile
from PIL import Image


def image_read(image_path, region=None):
    image_path = str(image_path)

    if image_path.endswith(("jpg", "png")):
        if not is_single_channel(image_path):
            raise Exception("Not supported image format. Movie files are not supported.")
            return None
        try:
            img = imageio.imread(image_path, pilmode='L', as_gray=True)
            img = img.astype(np.uint8)
        except ValueError:
            sys.exit("Image "+image_path+" is not valid. Please check out.")
    elif image_path.endswith(("mrc", "mrcs")):
        print(image_path)
        if not is_single_channel(image_path):
            raise Exception("Not supported image format. Movie files are not supported.")
            return None
        img = read_mrc(image_path)
    elif image_path.endswith(("tif", "tiff")):
        img = imageio.imread(image_path)
    else:
        raise Exception(image_path + "is not supported image format.")

    if np.issubdtype(img.dtype, np.uint32):
        img = img.astype(dtype=np.float64)

    if np.issubdtype(img.dtype, np.uint16):
        img = img.astype(dtype=np.float32)

    if np.issubdtype(img.dtype, np.float16):
        img = img.astype(dtype=np.float32)

    if region is not None:
        return img[region[1], region[0]]

    return img


def read_mrc(image_path):
    mrc_image_data = mrcfile.open(image_path, permissive=True, mode='r+')
    mrc_image_data = mrc_image_data.data
    mrc_image_data = np.squeeze(mrc_image_data)
    mrc_image_data = np.flipud(mrc_image_data)

    return mrc_image_data


def is_single_channel(image_path):
    if image_path.endswith(("mrc", "mrcs")):
        with mrcfile.mmap(image_path, permissive=True, mode='r+') as mrc:
            if mrc.header.nz > 1:
                return False
    if image_path.endswith(("tif", "tiff", "jpg", "png")):
        im = Image.open(image_path)
        if len(im.size) > 2:
            return False

    return True


def image_write(image_path, image):
    if image_path.endswith(("jpg", 'png')):
        imageio.imwrite(image_path, image)
    elif image_path.endswith(("tif", 'tiff')):
        image = np.float32(image)
        imageio.imwrite(image_path, image)
    elif image_path.endswith(("mrc", "mrcs")):
        image = np.flipud(image)
        with mrcfile.new(image_path, overwrite=True) as mrc:
            mrc.set_data(np.float32(image))


def read_width_height(image_path):
    width, height = 0, 0
    if image_path.endswith(("tif", "tiff", "jpg", "png")):
        im = Image.open(image_path)
        width, height = [int(i) for i in im.size]

    elif image_path.endswith(("mrc", "mrcs")):
        with mrcfile.mmap(image_path, permissive=True, mode='r') as mrc:
            width = mrc.header.ny
            height = mrc.header.nx

    return int(width), int(height)


def get_tile_coordinates(imgw, imgh, num_patches, patchxy, overlap=0):
    patch_width = int(imgw / num_patches)
    patch_height = int(imgh / num_patches)
    region_from_x = int(patchxy[0] * patch_width)
    region_to_x = int((patchxy[0] + 1) * patch_width)
    region_from_y = int(patchxy[1] * patch_height)
    region_to_y = int((patchxy[1] + 1) * patch_height)
    overlap = int(overlap)
    if patchxy[0] == 0:
        region_to_x = region_to_x + overlap * 2
    elif patchxy[0] == num_patches - 1:
        region_from_x = region_from_y - overlap * 2
    else:
        region_from_x = region_from_x - overlap
        region_to_x = region_to_x + overlap

    if patchxy[1] == 0:
        region_to_y = region_to_y + overlap * 2
    elif patchxy[1] == num_patches - 1:
        region_from_y = region_from_y - overlap * 2
    else:
        region_from_y = region_from_y - overlap
        region_to_y = region_to_y + overlap

    region_to_x = min(region_to_x, imgw)
    region_to_y = min(region_to_y, imgh)

    tile = np.s_[region_from_x:region_to_x, region_from_y:region_to_y]
    return tile