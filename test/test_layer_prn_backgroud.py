import unittest
import numpy as np

from MRCNN.config import Config
from MRCNN.data.dataset import Dataset
from MRCNN.data.meta_frcnn_data_loader import DataLoader
from MRCNN.enums import Mode
from MRCNN.layer import PrnBackground


class TestCast(unittest.TestCase):
    def setUp(self) -> None:
        self.train_json_path='/home/tmdocker/host/dataset/coco/annotations/instances_train2017.json'
        self.train_image_path='/home/tmdocker/host/dataset/coco/train2017/'
        self.val_json_path='/home/tmdocker/host/dataset/coco/annotations/instances_val2017.json'
        self.val_image_path='/home/tmdocker/host/dataset/coco/val2017/'


    def test_run_train_model(self):
        config = Config()
        novel_classes = (1,2,3)
        config.NUM_CLASSES = config.NUM_CLASSES - len(novel_classes)
        dataset = Dataset(json_path=self.val_json_path,
                          image_path=self.val_image_path)
        active_class_ids = [cat for cat in dataset.coco.cats if cat not in novel_classes]
        loader = DataLoader(config, Mode.TRAIN, 2, active_class_ids=active_class_ids,dataset=dataset, novel_classes=novel_classes)
        layer = PrnBackground(config)
        for i, data in enumerate(loader):
            resized_image, resized_boxes, dataloader_class_ids,rpn_match, rpn_bbox, active_class_ids, prn_images = data[0]
            output_prn_image = layer(resized_image, resized_boxes, prn_images)

            self.assertEqual(tuple(output_prn_image.numpy().shape[1:]), (config.NUM_CLASSES, *config.PRN_IMAGE_SIZE,4))

            bg_img = output_prn_image[0,0,...].numpy() # [224,224,4]
            bg_img = bg_img + np.array(tuple(config.MEAN_PIXEL[::-1])+(0,))
            bg_img = bg_img * (1,1,1,255)
            bg_img = bg_img.astype(np.uint8)
            if i==30:
                break