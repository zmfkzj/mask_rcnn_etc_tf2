import datetime
import imgaug.augmenters as iaa
import keras.api._v2.keras as keras
import numpy as np
import tensorflow as tf
import os

from MRCNN.config import Config
from MRCNN.data.frcnn_data_loader import DataLoader
from MRCNN.data.dataset_old import Dataset
from MRCNN.model import FasterRcnn
from MRCNN.enums import TrainLayers, Mode
from MRCNN.utils import LossWeight



config = Config(GPUS=0,
                ORIGIN_NUM_CLASSES=80+1,
                LEARNING_RATE=0.0001,
                TRAIN_IMAGES_PER_GPU=5,
                TEST_IMAGES_PER_GPU=10,
                STEPS_PER_EPOCH=2000,
                VALIDATION_STEPS=100,
                )

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

if not os.path.isdir(f'save_{now}/chpt'):
    os.makedirs(f'save_{now}/chpt')

with config.STRATEGY.scope():
    model = FasterRcnn(config)


###########################
# FPN+ train
###########################
train_loader = DataLoader(config, Mode.TRAIN, dataset=train_dataset,augmentations=augmentations, batch_size=12)
val_loader = DataLoader(config, Mode.TEST, dataset=val_dataset)

lr_schedule = CustomScheduler(config.LEARNING_RATE, 100*config.STEPS_PER_EPOCH,0.1,1, staircase=True)
with config.STRATEGY.scope():
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=config.GRADIENT_CLIP_NORM)
    model.compile(val_dataset, train_loader.active_class_ids,optimizer=optimizer, train_layers=TrainLayers.FPN_P)

callbacks = [keras.callbacks.ModelCheckpoint(f'save_{now}/chpt/fpn_p/best.h5',monitor='val_mAP50',save_best_only=True, save_weights_only=True,mode='max'),
            keras.callbacks.TensorBoard(log_dir=f'save_{now}/logs/fpn_p'),
            keras.callbacks.EarlyStopping('val_mAP50',patience=10,verbose=1, mode='max',restore_best_weights=True)]

model.fit(iter(train_loader), 
        epochs=1,
        callbacks=callbacks,
        validation_data=iter(val_loader), 
        steps_per_epoch=config.STEPS_PER_EPOCH,
        validation_steps=config.VALIDATION_STEPS)


###########################
# finetune
###########################
train_loader = DataLoader(config, Mode.TRAIN, dataset=train_dataset,augmentations=augmentations)
val_loader = DataLoader(config, Mode.TEST, dataset=val_dataset)

lr_schedule = CustomScheduler(config.LEARNING_RATE/10, 100*config.STEPS_PER_EPOCH,0.1,1, staircase=True)
with config.STRATEGY.scope():
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=config.GRADIENT_CLIP_NORM)
    model.compile(val_dataset, train_loader.active_class_ids,optimizer=optimizer, train_layers=TrainLayers.ALL)

callbacks = [keras.callbacks.ModelCheckpoint(f'save_{now}/chpt/all/best.h5',monitor='val_mAP50',save_best_only=True, save_weights_only=True,mode='max'),
            keras.callbacks.TensorBoard(log_dir=f'save_{now}/logs/all'),
            keras.callbacks.EarlyStopping('val_mAP50',patience=10,verbose=1, mode='max',restore_best_weights=True)]

model.fit(iter(train_loader), 
        epochs=300000,
        callbacks=callbacks,
        validation_data=iter(val_loader), 
        steps_per_epoch=config.STEPS_PER_EPOCH,
        validation_steps=config.VALIDATION_STEPS)


result = model.evaluate(iter(val_loader),steps=500)
print(result)
