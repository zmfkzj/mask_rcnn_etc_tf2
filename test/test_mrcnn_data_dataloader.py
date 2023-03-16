import random
import unittest

import cv2
import imgaug.augmenters as iaa
import numpy as np
import tensorflow as tf

from MRCNN.config import Config
from MRCNN.data.dataset import Dataset
from MRCNN.data.mrcnn_data_loader import DataLoader
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
        active_class_ids = [cat['id'] for cat in dataset.coco.dataset['categories']]
        loader = DataLoader(config,Mode.PREDICT,
                            image_pathes=['/home/tmdocker/host/dataset/5_coco_merge/images/task_221227 옥천교 gt 1024x1024 분할-2023_01_10_16_06_32-coco 1.0/', 
                                          '/home/tmdocker/host/dataset/5_coco_merge/images/task_금빛노을교 3차 1024x1024 분할 gt-2022_12_14_11_47_56-coco 1.0/'])
        loader = iter(loader)
        print(next(loader))
        print(next(loader))
        print(next(loader))


    def test_make_Dataloader_rcnn_test(self):
        config = Config()
        dataset = Dataset('/home/tmdocker/host/dataset/5_coco_merge/annotations/instances_test.json', 
                          '/home/tmdocker/host/dataset/5_coco_merge/images')
        active_class_ids = [cat['id'] for cat in dataset.coco.dataset['categories']]
        loader = DataLoader(config, Mode.TEST, dataset=dataset)
        loader = iter(loader)
        print(next(loader))
        print(next(loader))
        print(next(loader))


    def test_make_Dataloader_rcnn_train(self):
        config = Config()
        dataset = Dataset('/home/tmdocker/host/dataset/5_coco_merge/annotations/instances_test.json', 
                          '/home/tmdocker/host/dataset/5_coco_merge/images')
        active_class_ids = [cat['id'] for cat in dataset.coco.dataset['categories']]
        augmentations = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.GaussianBlur(),
            # iaa.Add(per_channel=True),
            # iaa.Multiply(per_channel=True),
            # iaa.GammaContrast(per_channel=True)
        ])
        loader = DataLoader(config, Mode.TRAIN, dataset=dataset,augmentations=augmentations)
        loader = iter(loader)
        i = next(loader)
        i = next(loader)
        i = next(loader)
    

    def test_build_rpn_targets(self):
        config = Config()
        dataset = Dataset(self.val_json_path, 
                          self.val_image_path)
        active_class_ids = [cat['id'] for cat in dataset.coco.dataset['categories']]
        loader = DataLoader(config, Mode.TRAIN, dataset=dataset)
        for i, data in enumerate(loader):
            resized_image, resized_boxes, minimize_masks, dataloader_class_ids,rpn_match, rpn_bbox, active_class_ids = data[0]
            dataloader_class_ids = tf.squeeze(dataloader_class_ids, 0)
            resized_boxes = tf.squeeze(resized_boxes,0)

            idx = tf.squeeze(tf.where(dataloader_class_ids>0),1)
            dataloader_class_ids = tf.gather(dataloader_class_ids, idx)
            resized_boxes = tf.gather(resized_boxes, idx)
            outputs = loader.build_rpn_targets(dataloader_class_ids, resized_boxes)

            if i==10:
                break


    def test_gt_image_match(self):
        config = Config()
        dataset = Dataset(self.val_json_path, 
                          self.val_image_path)
        active_class_ids = [cat['id'] for cat in dataset.coco.dataset['categories']]
        augmentations = iaa.Sequential([
            iaa.Fliplr(0.5),
        ])
        loader = DataLoader(config, Mode.TRAIN, dataset, augmentations=augmentations)
        # loader = DataLoader(config, Mode.TRAIN, 1, active_class_ids, dataset)
        for i, data in enumerate(loader):
            resized_image, resized_boxes, minimize_masks, dataloader_class_ids,rpn_match, rpn_bbox, active_class_ids = data[0]
            img = resized_image[0].numpy()[...,::-1]
            img = np.ascontiguousarray(img, dtype=np.uint8)
            boxes = np.ascontiguousarray(resized_boxes[0].numpy())
            masks = np.ascontiguousarray(minimize_masks[0].numpy()).transpose([2,0,1])
            full_masks = []
            for box,mask in zip(boxes, masks):
                if np.any(box!=0):
                    y1,x1,y2,x2 = np.around(box).astype(np.int32)
                    mask = mask.astype(np.uint8)
                    full_mask = unmold_mask(mask,box,config.IMAGE_SHAPE)
                    full_masks.append(full_mask)
                    img = cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255))
            full_mask = np.any(full_masks,axis=0).astype(np.uint8)
            full_mask = np.broadcast_to(np.expand_dims(full_mask, -1),[*config.IMAGE_SHAPE])*(0,255,0)
            img = cv2.addWeighted(img,0.6, full_mask.astype(np.uint8),0.4,0)

            if i==10:
                break


