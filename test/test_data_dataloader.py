import random
import unittest
from MRCNN.config import Config
from MRCNN.data.data_loader import DataLoader, Mode
from MRCNN.data.dataset import Dataset
import tensorflow as tf
import imgaug.augmenters as iaa
from pydantic.dataclasses import dataclass

from MRCNN.utils import compute_backbone_shapes, generate_pyramid_anchors


# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()

class TestDataLoader(unittest.TestCase):
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