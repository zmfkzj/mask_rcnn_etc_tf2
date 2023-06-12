
import unittest
from MRCNN.data.dataset import Dataset
from MRCNN.data.frcnn_data_loader import *
import datetime
import sys
sys.setrecursionlimit(10**6)

class TestDataDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.json_path = '/home/min/4t/dataset/coco/annotations/instances_val2017.json'

    def test_measure_time_dataloader(self):
        dataset = Dataset.from_json(self.json_path,'/home/min/4t/dataset/coco/val2017/')

        config = Config(GPUS=0,
                        LEARNING_RATE=0.0001,
                        TRAIN_IMAGES_PER_GPU=3,
                        TEST_IMAGES_PER_GPU=10,
                        STEPS_PER_EPOCH=2000,
                        VALIDATION_STEPS=200
                        )

        loader = make_train_dataloader(dataset, config, 40)
        
        now = datetime.datetime.now()
        for i, _ in enumerate(loader):
            time = datetime.datetime.now() - now
            now = datetime.datetime.now()
            print(time)
            if i > 10:
                break
