import random
import unittest
import numpy as np
from MRCNN.config import Config
from MRCNN.data.data_loader import DataLoader, Mode
from MRCNN.data.dataset import Dataset
import tensorflow as tf
import imgaug.augmenters as iaa
from pydantic.dataclasses import dataclass
import cv2

from MRCNN.utils import compute_backbone_shapes, generate_pyramid_anchors, unmold_mask


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
        loader = DataLoader(config, active_class_ids,Mode.PREDICT, 2,
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
        loader = DataLoader(config, active_class_ids, Mode.TEST, 2, dataset=dataset)
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
        loader = DataLoader(config, Mode.TRAIN, 2,active_class_ids=active_class_ids, dataset=dataset,augmentations=augmentations)
        loader = iter(loader)
        print(next(loader))
        print(next(loader))
        print(next(loader))
    

    def test_build_rpn_targets(self):
        class TestDataLoader(DataLoader):
            def __post_init__(self):
                backbone_shapes = compute_backbone_shapes(self.config)
                self.anchors = generate_pyramid_anchors(
                                    self.config.RPN_ANCHOR_SCALES,
                                    self.config.RPN_ANCHOR_RATIOS,
                                    backbone_shapes,
                                    self.config.BACKBONE_STRIDES,
                                    self.config.RPN_ANCHOR_STRIDE)

        config = Config()
        config.BACKBONE_STRIDES = [2, 4, 8, 16]
        config.RPN_ANCHOR_SCALES = (16, 32, 64, 128)
        dataset = Dataset('/home/tmdocker/host/dataset/5_coco_merge/annotations/instances_test.json', 
                          '/home/tmdocker/host/dataset/5_coco_merge/images')
        active_class_ids = [cat['id'] for cat in dataset.coco.dataset['categories']]
        loader = TestDataLoader(config, Mode.TRAIN, 2, active_class_ids=active_class_ids, dataset=dataset)
        dataloader_class_ids = tf.constant(random.choices(active_class_ids,k=100))
        boxes = tf.ones([100,4])
        loader.build_rpn_targets(dataloader_class_ids, boxes)


    def test_processing_train(self):
        class TestDataLoader(DataLoader):
            def __post_init__(self):
                backbone_shapes = compute_backbone_shapes(self.config)
                self.anchors = generate_pyramid_anchors(
                                    self.config.RPN_ANCHOR_SCALES,
                                    self.config.RPN_ANCHOR_RATIOS,
                                    backbone_shapes,
                                    self.config.BACKBONE_STRIDES,
                                    self.config.RPN_ANCHOR_STRIDE)

        config = Config()
        dataset = Dataset('/home/tmdocker/host/dataset/5_coco_merge/annotations/instances_test.json', 
                          '/home/tmdocker/host/dataset/5_coco_merge/images')
        active_class_ids = [cat['id'] for cat in dataset.coco.dataset['categories']]
        loader = TestDataLoader(config, active_class_ids, Mode.TRAIN, 2, dataset=dataset, shuffle_buffer_size=16)
        loader.preproccessing_train(dataset.coco.dataset['images'][0]['path'], dataset.coco.getAnnIds(dataset.coco.dataset['images'][0]['id']))
    

    def test_gt_image_match(self):
        config = Config()
        dataset = Dataset(self.val_json_path, 
                          self.val_image_path)
        active_class_ids = [cat['id'] for cat in dataset.coco.dataset['categories']]
        augmentations = iaa.Sequential([
            iaa.Fliplr(0.5),
        ])
        loader = DataLoader(config, Mode.TRAIN, 1, active_class_ids, dataset, augmentations=augmentations)
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


