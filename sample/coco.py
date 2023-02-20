import datetime
import imgaug.augmenters as iaa
import keras.api._v2.keras as keras
import numpy as np
import tensorflow as tf

from MRCNN.config import Config
from MRCNN.data.data_loader import DataLoader, Mode
from MRCNN.data.dataset import Dataset
from MRCNN.metric import CocoMetric, EvalType
from MRCNN.model.mask_rcnn import MaskRcnn, TrainLayers



class TrainConfig(Config):
    GPUS = 0,1
    # GPUS = 0
    NUM_CLASSES = 1+80 
    LEARNING_RATE = 0.0005
    TRAIN_IMAGES_PER_GPU = 12
    TEST_IMAGES_PER_GPU = 20
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 50
    

config = TrainConfig()

class CustomScheduler(keras.optimizers.schedules.ExponentialDecay):
    def __init__(self, initial_learning_rate, decay_steps, decay_rate, burnin_step, staircase=False, name=None):
        super().__init__(initial_learning_rate, decay_steps, decay_rate, staircase, name)
        self.burnin_step = burnin_step
        self.initial_learning_rate = initial_learning_rate

    def __call__(self, step):
        super_lr = super().__call__(step)
        return tf.cond(step<=self.burnin_step, 
                lambda : tf.cast(self.initial_learning_rate*tf.math.pow(step/self.burnin_step,4),tf.float32),
                lambda : super_lr)

lr_schedule = CustomScheduler(config.LEARNING_RATE, 9000,0.95,500, staircase=True)

augmentations = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(),
    # iaa.Add(per_channel=True),
    # iaa.Multiply(per_channel=True),
    # iaa.GammaContrast(per_channel=True)
])

now = datetime.datetime.now().isoformat()
train_dataset = Dataset('/home/tmdocker/host/dataset/coco/annotations/instances_train2017.json', 
                    '/home/tmdocker/host/dataset/coco/train2017/')
val_dataset = Dataset('/home/tmdocker/host/dataset/coco/annotations/instances_val2017.json', 
                    '/home/tmdocker/host/dataset/coco/val2017/')

active_class_ids = [cat['id'] for cat in train_dataset.coco.dataset['categories']]

train_loader = DataLoader(config, Mode.TRAIN, config.TRAIN_BATCH_SIZE, active_class_ids=active_class_ids, dataset=train_dataset,augmentations=augmentations)
val_loader = DataLoader(config, Mode.TEST,config.TEST_BATCH_SIZE, active_class_ids=active_class_ids, dataset=val_dataset)

callbacks = [keras.callbacks.ModelCheckpoint(f'save_{now}',monitor='mAP50',save_best_only=True, save_weights_only=True),
             keras.callbacks.TensorBoard(log_dir=f'logs_{now}',update_freq='s'),
             keras.callbacks.EarlyStopping('mAP50')]

val_metric = CocoMetric(val_dataset, config, active_class_ids,eval_type=EvalType.SEGM)

with config.STRATEGY.scope():
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=config.GRADIENT_CLIP_NORM)
    model = MaskRcnn(config)

    model.set_trainable(TrainLayers.HEADS)
    model.compile(val_metric,optimizer=optimizer)

model.fit(iter(train_loader), 
          epochs=100,
          callbacks=callbacks,
          validation_data=iter(val_loader), 
          steps_per_epoch=config.STEPS_PER_EPOCH,
          validation_steps=config.VALIDATION_STEPS,
          validation_freq=10)


with config.STRATEGY.scope():
    model.set_trainable(TrainLayers.ALL)
    model.compile(val_metric,optimizer=optimizer)

model.fit(iter(train_loader), 
          epochs=300,
          callbacks=callbacks,
          validation_data=iter(val_loader), 
          steps_per_epoch=config.STEPS_PER_EPOCH,
          validation_steps=config.VALIDATION_STEPS,
          validation_freq=10)

model.evaluate(iter(val_loader))