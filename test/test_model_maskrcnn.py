import unittest

import numpy as np
from MRCNN.config import Config
from MRCNN.data.data_loader import DataLoader, Mode
from MRCNN.metric import CocoMetric, EvalType
from MRCNN.model import MaskRcnn
from MRCNN.data.dataset import Dataset

class TestModel(unittest.TestCase):
    def test_make_model(self):
        config = Config()
        dataset = Dataset(json_path='/home/tmdocker/host/dataset/5_coco_merge/annotations/instances_train.json',
                          image_path='/home/tmdocker/host/dataset/5_coco_merge/images')
        model = MaskRcnn(config, dataset)
        active_class_ids = [cat for cat in dataset.coco.cats]
        coco_metric = CocoMetric(dataset, config, active_class_ids,iou_thresh=0.5, eval_type=EvalType.SEGM)
        model.compile(coco_metric)
        self.assertTrue(True)

    def test_run_train_model(self):
        config = Config()
        dataset = Dataset(json_path='/home/tmdocker/host/dataset/5_coco_merge/annotations/instances_train.json',
                          image_path='/home/tmdocker/host/dataset/5_coco_merge/images')
        active_class_ids = [cat['id'] for cat in dataset.coco.dataset['categories']]
        loader = DataLoader(config, active_class_ids, Mode.TRAIN, 2, dataset=dataset, shuffle_buffer_size=16)
        model = MaskRcnn(config, dataset)
        for data in loader:
            resized_image, resized_boxes, minimize_masks, dataloader_class_ids,rpn_match, rpn_bbox, active_class_ids = data[0]
            losses = \
                model.train_model([resized_image, resized_boxes, minimize_masks, dataloader_class_ids,rpn_match, rpn_bbox, active_class_ids])
            print(losses)
            break
        

