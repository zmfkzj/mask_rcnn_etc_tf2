import tensorflow as tf
import tensorflow.keras as keras
import imgaug.augmenters as iaa

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from MRCNN.data.data_loader import CocoDataset
from MRCNN import MaskRCNN, Trainer, Detector, Evaluator
from MRCNN.config import Config


class TrainConfig(Config):
    GPUS = 0,1
    # GPUS = 0
    NUM_CLASSES = 1+80 
    LEARNING_RATE = 0.001
    IMAGES_PER_GPU = 3

train_dataset = CocoDataset()
train_dataset.load_coco('d:/coco/train2017/', 'd:/coco/annotations/instances_train2017.json')

train_config = TrainConfig()
model = MaskRCNN(train_config)

lr_schedule = keras.optimizers.schedules.ExponentialDecay(train_config.LEARNING_RATE/train_config.STEPS_PER_EPOCH, 40000, 0.1, staircase=True,)

optimizer = keras.optimizers.SGD(learning_rate=lr_schedule,
                                momentum=train_config.LEARNING_MOMENTUM)

ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
manager = tf.train.CheckpointManager(ckpt, directory='save_model', max_to_keep=None)
# status = ckpt.restore('save_model/ckpt-1')
status = ckpt.restore(manager.latest_checkpoint)

val_evaluator = Evaluator(model, 'C:/coco/val2017/', 'C:/coco/annotations/instances_val2017.json',train_config, conf_thresh=0.25)
augmentation = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(),
    iaa.Add(per_channel=True),
    iaa.Multiply(per_channel=True),
    iaa.GammaContrast(per_channel=True)

])
trainer = Trainer(model, train_dataset, manager, config=train_config, optimizer=optimizer, val_evaluator=val_evaluator)
trainer.train(160, 'all')
trainer.train(90, '4+')
trainer.train(40, 'heads')

class ValConfig(Config):
    GPUS = 0,1
    # GPUS = 0
    NUM_CLASSES = 1+80 
    IMAGES_PER_GPU = 20

val_config = ValConfig()
model = MaskRCNN(val_config)

ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
manager = tf.train.CheckpointManager(ckpt, directory='save_model', max_to_keep=None)
status = ckpt.restore(manager.latest_checkpoint)

val_evaluator = Evaluator(model, 'C:/coco/val2017/', 'C:/coco/annotations/instances_val2017.json',val_config, conf_thresh=0.25)
metric = val_evaluator.eval('d:/')
print(metric)

# classes = [info['name'] for info in train_dataset.class_info]
# detector = Detector(model, classes, config)
# detections = detector.detect('d:/coco/val2017/', shuffle=True, limit_step=10)
