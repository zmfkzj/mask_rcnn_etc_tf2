import unittest

import tensorflow as tf
import numpy as np
from MRCNN.config import Config
from MRCNN.data.data_loader import DataLoader, Mode
from MRCNN.metric import CocoMetric, EvalType
from MRCNN.model import MaskRcnn
from MRCNN.data.dataset import Dataset


# tf.config.run_functions_eagerly(True)

class TestModel(unittest.TestCase):
    def test_make_model(self):
        config = Config()
        dataset = Dataset(json_path='/home/tmdocker/host/dataset/5_coco_merge/annotations/instances_train.json',
                          image_path='/home/tmdocker/host/dataset/5_coco_merge/images')
        model = MaskRcnn(config)
        active_class_ids = [cat for cat in dataset.coco.cats]
        coco_metric = CocoMetric(dataset, config, active_class_ids,iou_thresh=0.5, eval_type=EvalType.SEGM)
        model.compile(coco_metric)
        self.assertTrue(True)

    def test_run_train_model(self):
        config = Config()
        dataset = Dataset(json_path='/home/tmdocker/host/dataset/5_coco_merge/annotations/instances_train.json',
                          image_path='/home/tmdocker/host/dataset/5_coco_merge/images')
        active_class_ids = [cat['id'] for cat in dataset.coco.dataset['categories']]
        loader = DataLoader(config, Mode.TRAIN, 2, active_class_ids=active_class_ids,dataset=dataset, shuffle_buffer_size=16)
        model = MaskRcnn(config)
        for data in loader:
            resized_image, resized_boxes, minimize_masks, dataloader_class_ids,rpn_match, rpn_bbox, active_class_ids = data[0]
            losses = \
                model.train_model([resized_image, resized_boxes, minimize_masks, dataloader_class_ids,rpn_match, rpn_bbox, active_class_ids])
            print(losses)
            break
        

    def test_run_test_model(self):
        config = Config()
        dataset = Dataset(json_path='/home/tmdocker/host/dataset/5_coco_merge/annotations/instances_train.json',
                          image_path='/home/tmdocker/host/dataset/5_coco_merge/images')
        active_class_ids = [cat['id'] for cat in dataset.coco.dataset['categories']]
        loader = DataLoader(config, Mode.TEST, 2, active_class_ids=active_class_ids, dataset=dataset, shuffle_buffer_size=16)
        model = MaskRcnn(config).test_model
        for data in loader:
            input_images, input_window, origin_image_shapes, image_ids = data[0]
            results = model([input_images, input_window])
            print(results)
            break

    def test_run_predict_model(self):
        config = Config()
        loader = DataLoader(config, Mode.PREDICT, 2, image_pathes='/home/tmdocker/host/dataset/5_coco_merge/images/task_221227 옥천교 gt 1024x1024 분할-2023_01_10_16_06_32-coco 1.0/')
        model = MaskRcnn(config).predict_model
        for data in loader:
            input_images, input_window, origin_image_shapes, pathes = data[0]
            results = model([input_images, input_window])
            print(results)
            break
    
    def test_train(self):
        config = Config()
        train_dataset = Dataset('/home/tmdocker/host/dataset/coco/annotations/instances_train2017.json', 
                            '/home/tmdocker/host/dataset/coco/train2017/')
        val_dataset = Dataset('/home/tmdocker/host/dataset/coco/annotations/instances_val2017.json', 
                            '/home/tmdocker/host/dataset/coco/val2017/')

        active_class_ids = [cat['id'] for cat in train_dataset.coco.dataset['categories']]

        train_loader = DataLoader(config, Mode.TRAIN, config.BATCH_SIZE, active_class_ids=active_class_ids, dataset=train_dataset, shuffle_buffer_size=16)
        val_loader = DataLoader(config, Mode.TEST, config.TEST_BATCH_SIZE, active_class_ids=active_class_ids, dataset=val_dataset)
        val_metric = CocoMetric(val_dataset, config, active_class_ids,eval_type=EvalType.SEGM)

        with config.STRATEGY.scope():
            model = MaskRcnn(config)
            model.compile(val_metric,optimizer='adam')

        model.fit(iter(train_loader), epochs=2,validation_data=iter(val_loader), steps_per_epoch=2,validation_steps=2)


    def test_evaluate(self):
        config = Config()
        val_dataset = Dataset('/home/tmdocker/host/dataset/coco/annotations/instances_val2017.json', 
                            '/home/tmdocker/host/dataset/coco/val2017/')
        active_class_ids = [cat['id'] for cat in val_dataset.coco.dataset['categories']]
        val_loader = DataLoader(config, Mode.TEST, config.TEST_BATCH_SIZE, active_class_ids=active_class_ids, dataset=val_dataset)
        val_metric = CocoMetric(val_dataset, config, active_class_ids,eval_type=EvalType.SEGM)

        with config.STRATEGY.scope():
            model = MaskRcnn(config)
            model.compile(val_metric,optimizer='adam')

        results = model.evaluate(iter(val_loader), steps=2)
        print(results)


    def test_predict(self):
        config = Config()
        loader = DataLoader(config, Mode.PREDICT, 2, image_pathes='/home/tmdocker/host/dataset/5_coco_merge/images/task_221227 옥천교 gt 1024x1024 분할-2023_01_10_16_06_32-coco 1.0/')
        with config.STRATEGY.scope():
            model = MaskRcnn(config)
        results = model.predict(iter(loader),steps=2)
        print(results)