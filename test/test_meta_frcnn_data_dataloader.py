import random
import unittest

import cv2
import imgaug.augmenters as iaa
import numpy as np
import tensorflow as tf

from MRCNN.config import Config
from MRCNN.data.dataset import Dataset
from MRCNN.data.meta_frcnn_data_loader import DataLoader
from MRCNN.enums import Mode
from MRCNN.utils import (compute_backbone_shapes, generate_pyramid_anchors,
                         unmold_mask)

# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()

class TestDataLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.train_json_path='/home/tmdocker/host/dataset/coco/annotations/instances_train2017.json'
        self.train_image_path='/home/tmdocker/host/dataset/coco/train2017/'
        self.val_json_path='/home/tmdocker/host/dataset/coco/annotations/instances_val2017.json'
        self.val_image_path='/home/tmdocker/host/dataset/coco/val2017/'


    def test_make_Dataloader_rcnn_predict(self):
        config = Config()
        dataset = Dataset('/home/tmdocker/host/dataset/5_coco_merge/annotations/instances_test.json', 
                          '/home/tmdocker/host/dataset/5_coco_merge/images')
        loader = DataLoader(config, Mode.PREDICT, 2,
                            novel_classes=(1,2),
                            image_pathes=['/home/tmdocker/host/dataset/5_coco_merge/images/task_221227 옥천교 gt 1024x1024 분할-2023_01_10_16_06_32-coco 1.0/', 
                                          '/home/tmdocker/host/dataset/5_coco_merge/images/task_금빛노을교 3차 1024x1024 분할 gt-2022_12_14_11_47_56-coco 1.0/'])
        loader = iter(loader)
        print(next(loader))
        print(next(loader))
        print(next(loader))


    def test_make_Dataloader_rcnn_test(self):
        config = Config()
        dataset = Dataset(self.val_json_path, 
                          self.val_image_path)
        active_class_ids = [cat['id'] for cat in dataset.coco.dataset['categories']]
        loader = DataLoader(config, Mode.TEST, 2, active_class_ids=active_class_ids ,dataset=dataset, novel_classes=(1,2,3))
        loader = iter(loader)
        print(next(loader))
        print(next(loader))
        print(next(loader))


    def test_make_Dataloader_rcnn_train(self):
        config = Config()
        dataset = Dataset(self.val_json_path, 
                          self.val_image_path)
        active_class_ids = [cat['id'] for cat in dataset.coco.dataset['categories']]
        augmentations = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.GaussianBlur(),
            # iaa.Add(per_channel=True),
            # iaa.Multiply(per_channel=True),
            # iaa.GammaContrast(per_channel=True)
        ])
        loader = DataLoader(config, Mode.TRAIN, 2,
                            novel_classes=(1,2),
                            active_class_ids=active_class_ids, 
                            dataset=dataset,
                            augmentations=augmentations)
        loader = iter(loader)
        print(next(loader))
        print(next(loader))
        print(next(loader))
    


    def test_prn_image_match(self):
        config = Config()
        dataset = Dataset(self.val_json_path, 
                          self.val_image_path)
        active_class_ids = [cat['id'] for cat in dataset.coco.dataset['categories']]
        augmentations = iaa.Sequential([
            iaa.Fliplr(0.5),
        ])
        loader = DataLoader(config, Mode.TRAIN, 1,
                            active_class_ids=active_class_ids, 
                            novel_classes= (1,2,3),
                            phase= 1,
                            dataset=dataset, 
                            augmentations=augmentations)
        for i, data in enumerate(loader):
            resized_image, resized_boxes, dataloader_class_ids,rpn_match, rpn_bbox, active_class_ids, prn_images = data[0]
            imgs = prn_images[0].numpy()
            for j, img in enumerate(imgs):
                img = img + np.concatenate([config.MEAN_PIXEL,[0]])
                img = img.astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
                img = np.ascontiguousarray(img, dtype=np.uint8)

                if j==30:
                    break
            break


