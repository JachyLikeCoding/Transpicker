# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from torchvision
# ------------------------------------------------------------------------

"""
Copy-Paste from torchvision, but add utility of caching images on memory
Add read mrc files ----Chier
"""
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
import tqdm
import numpy as np
import cv2
from io import BytesIO
from transpicker.read_image import read_mrc, image_read


class CocoDetection(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None,
                 cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cache_mode = cache_mode
        self.local_rank = local_rank
        self.local_size = local_size
        if cache_mode:
            self.cache = {}
            self.cache_images()

    def cache_images(self):
        self.cache = {}
        for index, img_id in zip(tqdm.trange(len(self.ids)), self.ids):
            if index % self.local_size != self.local_rank:
                continue
            path = self.coco.loadImgs(img_id)[0]['file_name']
            with open(os.path.join(self.root, path), 'rb') as f:
                self.cache[path] = f.read()

    def get_image(self, path):
        # # TODO: 这里测试用，先直接加一个
        # print("read image path: ", os.path.join(self.root, path))
        # image = read_mrc(os.path.join(self.root, path))
        # print('read mrc image:',image)
        # cvimage = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # print('cvimage---------------------------------------------------test')
        # print('cvimage: ', cvimage)
        # print(cvimage.shape)

        # return cvimage

        if self.cache_mode:
            if path not in self.cache.keys():
                with open(os.path.join(self.root, path), 'rb') as f:
                    self.cache[path] = f.read()
            # for cryodata datasets, read mrc file data directly.
            # if path.endswith('.mrc'):
            #     image = read_mrc(BytesIO(self.cache[path]))
            #     # 需要转成PIL
            #     im = Image.fromarray(image)
            #     # im = np.expand_dims(im, 3)
            #     return im
            return Image.open(BytesIO(self.cache[path])).convert('RGB')

        # if path.endswith('.mrc'):
        #     image = read_mrc(os.path.join(self.root, path))
        #     cvimage = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        #     print('cvimage---------------------------------------------------test')
        #     # print(cvimage)
        #     # im = np.asarray(cvimage)
        #     # im = Image.fromarray(image)
        #     # im = np.expand_dims(im, 3)
        #     return cvimage
        return Image.open(os.path.join(self.root, path)).convert('RGB')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = self.get_image(path)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)
