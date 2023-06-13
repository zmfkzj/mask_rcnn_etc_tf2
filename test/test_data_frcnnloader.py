
import unittest
from MRCNN.data.dataset import Dataset
from MRCNN.data.frcnn_data_loader import *
import datetime
import sys
sys.setrecursionlimit(10**6)

class TestDataDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.json_path = '/home/min/4t/dataset/coco/annotations/instances_val2017.json'
        self.img_dir ='/home/min/4t/dataset/coco/val2017/' 

    def test_measure_time_dataloader(self):
        dataset = Dataset.from_json(self.json_path,self.img_dir)

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

    def test_check_dataloader_output(self):
        dataset = Dataset.from_json(self.json_path,self.img_dir)

        config = Config()

        loader = make_train_dataloader(dataset, config, 1)
        
        for i, data in enumerate(loader):
            input_images, input_gt_boxes, input_gt_class_ids, input_rpn_match, input_rpn_bbox, input_gt_masks= data

            img = input_images[0].numpy().copy()
            img = img * config.PIXEL_STD + config.PIXEL_MEAN
            img = cv2.cvtColor(img.round().astype(np.uint8), cv2.COLOR_RGB2BGR)
            for box in input_gt_boxes[0].numpy():
                y1,x1,y2,x2 = box.round().astype(np.int32)
                img = cv2.rectangle(img, (x1,y1),(x2,y2), color=(0,0,255))
            
            cv2.imwrite(f'{i}.jpg', img)

            if i == 30:
                break
