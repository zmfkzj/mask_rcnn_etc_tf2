import unittest

import numpy as np
from MRCNN.config import Config
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
        model.compile(coco_metric, jit_compile=True)
        self.assertTrue(True)
