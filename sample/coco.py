import tensorflow as tf
import tensorflow.keras as keras

from MRCNN.data.data_loader import CocoDataset
from MRCNN.model.mask_rcnn import MaskRCNN
from MRCNN.config import Config
from MRCNN.trainer import Trainer
from MRCNN.detector import Detector

class CustomConfig(Config):
    GPUS = 0,1
    # GPUS = 0
    NUM_CLASSES = 1+80 
    LEARNING_RATE = 0.0001
    IMAGES_PER_GPU = 3

# train_dataset = CocoDataset()
# train_dataset.load_coco('d:/coco/train2017/', 'd:/coco/annotations/instances_train2017.json')

val_dataset = CocoDataset()
val_dataset.load_coco('d:/coco/val2017/', 'd:/coco/annotations/instances_val2017.json')

config = CustomConfig()
model = MaskRCNN(config)

optimizer = keras.optimizers.SGD(learning_rate=config.LEARNING_RATE,
                                momentum=config.LEARNING_MOMENTUM,
                                decay=config.WEIGHT_DECAY)

ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
manager = tf.train.CheckpointManager(ckpt, directory='save_model', max_to_keep=None)
# status = ckpt.restore(manager.latest_checkpoint)
status = ckpt.restore('save_model/ckpt-50')
# trainer = Trainer(model, train_dataset, manager, config=config, optimizer=optimizer)
# trainer.train(50, 'all')

val_pathes = [val_dataset.image_info[id]['path'] for id in val_dataset.image_ids]
val_pathes = [info['path'] for info in val_dataset.image_info]
detector = Detector(model, config)
# detections = detector.detect(val_pathes, shuffle=True, limit_step=config.VALIDATION_STEPS)
detections = detector.detect(val_pathes, shuffle=True, limit_step=2)
print(detections)
