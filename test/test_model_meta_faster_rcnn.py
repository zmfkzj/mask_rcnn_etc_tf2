import unittest

from MRCNN.config import Config
from MRCNN.data.meta_frcnn_data_loader import DataLoader
from MRCNN.data.dataset import Dataset
from MRCNN.enums import EvalType, Mode
from MRCNN.model import MetaFasterRcnn

# tf.config.run_functions_eagerly(True)

class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        self.train_json_path='/home/tmdocker/host/dataset/coco/annotations/instances_train2017.json'
        self.train_image_path='/home/tmdocker/host/dataset/coco/train2017/'
        self.val_json_path='/home/tmdocker/host/dataset/coco/annotations/instances_val2017.json'
        self.val_image_path='/home/tmdocker/host/dataset/coco/val2017/'


    def test_make_model(self):
        config = Config()
        dataset = Dataset(json_path=self.train_json_path,
                          image_path=self.train_image_path)
        model = MetaFasterRcnn(config)
        active_class_ids = [cat for cat in dataset.coco.cats]
        model.compile(dataset, active_class_ids,optimizer='adam')
        self.assertTrue(True)

    def test_run_train_model(self):
        config = Config()
        novel_classes = (1,2,3)
        config.NUM_CLASSES = config.NUM_CLASSES - len(novel_classes)
        dataset = Dataset(json_path=self.val_json_path,
                          image_path=self.val_image_path)
        active_class_ids = [cat for cat in dataset.coco.cats if cat not in novel_classes]
        loader = DataLoader(config, Mode.TRAIN, 2, active_class_ids=active_class_ids,dataset=dataset, novel_classes=novel_classes)
        model = MetaFasterRcnn(config)
        for data in loader:
            resized_image, resized_boxes, dataloader_class_ids,rpn_match, rpn_bbox, active_class_ids, prn_images = data[0]
            losses = \
                model.train_model([resized_image, resized_boxes, dataloader_class_ids,rpn_match, rpn_bbox, active_class_ids, prn_images])
            print(losses)
            break
        

    def test_run_test_model(self):
        config = Config()
        dataset = Dataset(json_path=self.train_json_path,
                          image_path=self.train_image_path)
        active_class_ids = [cat for cat in dataset.coco.cats]
        loader = DataLoader(config, Mode.TEST, 2, active_class_ids=active_class_ids, dataset=dataset)
        model = MetaFasterRcnn(config).test_model
        for data in loader:
            input_images, input_window, origin_image_shapes, image_ids = data[0]
            results = model([input_images, input_window])
            print(results)
            break

    def test_run_predict_model(self):
        config = Config()
        loader = DataLoader(config, Mode.PREDICT, 2, image_pathes='/home/tmdocker/host/dataset/5_coco_merge/images/task_221227 옥천교 gt 1024x1024 분할-2023_01_10_16_06_32-coco 1.0/')
        model = MetaFasterRcnn(config).predict_model
        for data in loader:
            input_images, input_window, origin_image_shapes, pathes = data[0]
            results = model([input_images, input_window])
            print(results)
            break
    
    def test_train(self):
        novel_classes=(1,2,3)
        config = Config()
        config.GPUS = 0
        config.GPU_COUNT = 1
        config.NUM_CLASSES = config.NUM_CLASSES - len(novel_classes)
        train_dataset = Dataset(self.train_json_path, self.train_image_path)
        val_dataset = Dataset(self.val_json_path, self.val_image_path)

        active_class_ids = [cat for cat in train_dataset.coco.cats if cat not in novel_classes]

        train_loader = DataLoader(config, Mode.TRAIN, 2*config.GPU_COUNT, active_class_ids=active_class_ids, dataset=train_dataset, novel_classes=novel_classes)
        val_loader = DataLoader(config, Mode.TEST, 5*config.GPU_COUNT, active_class_ids=active_class_ids, dataset=val_dataset, novel_classes=novel_classes)

        with config.STRATEGY.scope():
            model = MetaFasterRcnn(config)
            model.compile(val_dataset, active_class_ids,optimizer='adam')

        hist = model.fit(iter(train_loader), epochs=2,validation_data=iter(val_loader), steps_per_epoch=2,validation_steps=2)
        print(hist.history)


    def test_evaluate(self):
        config = Config()
        val_dataset = Dataset(self.val_json_path, self.val_image_path)
        active_class_ids = [cat for cat in val_dataset.coco.cats]
        val_loader = DataLoader(config, Mode.TEST, config.TEST_BATCH_SIZE, active_class_ids=active_class_ids, dataset=val_dataset)

        with config.STRATEGY.scope():
            model = MetaFasterRcnn(config)
            model.compile(val_dataset, active_class_ids,optimizer='adam')

        results = model.evaluate(iter(val_loader), steps=2)
        print(results)


    def test_predict(self):
        config = Config()
        loader = DataLoader(config, Mode.PREDICT, 2, image_pathes='/home/tmdocker/host/dataset/5_coco_merge/images/task_221227 옥천교 gt 1024x1024 분할-2023_01_10_16_06_32-coco 1.0/')
        with config.STRATEGY.scope():
            model = MetaFasterRcnn(config)
        results = model.predict(iter(loader),steps=2)
        print(results)