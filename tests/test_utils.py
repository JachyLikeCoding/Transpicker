import cv2
import sys
sys.path.append("/home/zhangchi/transpicker/Transpicker")

from matplotlib import pyplot as plt
from transpicker import utils


if __name__ == "__main__":
    image_name = "/data/zhangchi/cryodata/empiar10017/train/Falcon_2012_06_12-14_33_35_0.jpg"
    image = cv2.imread(image_name)

    image_norm = utils.normalize(image)
    plt.imshow(image_norm)
    plt.show()

    image_norm_margin = utils.normalize(image, 0.1)
    plt.imshow(image_norm_margin)
    plt.show()

    image_norm_gmm = utils.normalize_gmm(image)
    plt.imshow(image_norm_gmm)
    plt.show()