import tensorflow as tf
import tensorflow.keras as keras

from MRCNN.data.data_loader import CocoDataset
from MRCNN import MaskRCNN, Trainer, Detector, Evaluator
from MRCNN.config import Config

class CustomConfig(Config):
    GPUS = 0,1
    # GPUS = 0
    NUM_CLASSES = 1+80 
    LEARNING_RATE = 0.001
    IMAGES_PER_GPU = 3

train_dataset = CocoDataset()
train_dataset.load_coco('d:/coco/train2017/', 'd:/coco/annotations/instances_train2017.json')

config = CustomConfig()
model = MaskRCNN(config)

lr_schedule = keras.optimizers.schedules.ExponentialDecay(config.LEARNING_RATE, 50000, 0.96, staircase=True)

optimizer = keras.optimizers.SGD(learning_rate=lr_schedule,
                                momentum=config.LEARNING_MOMENTUM)

ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
manager = tf.train.CheckpointManager(ckpt, directory='save_model', max_to_keep=None)
status = ckpt.restore(manager.latest_checkpoint)

val_evaluator = Evaluator(model, 'd:/coco/val2017/', 'd:/coco/annotations/instances_val2017.json',config)
trainer = Trainer(model, train_dataset, manager, config=config, optimizer=optimizer, val_evaluator=val_evaluator)
trainer.train(300, 'all')

# classes = [info['name'] for info in train_dataset.class_info]
# detector = Detector(model, classes, config)
# detections = detector.detect('d:/coco/val2017/', shuffle=True, limit_step=10)
