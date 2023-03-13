import datetime
import imgaug.augmenters as iaa
import keras.api._v2.keras as keras
import numpy as np
import tensorflow as tf
import os

# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)


from MRCNN.config import Config
from MRCNN.data.meta_frcnn_data_loader import DataLoader
from MRCNN.data.dataset import Dataset
from MRCNN.enums import EvalType, Model, TrainLayers, Mode
from MRCNN.model import MetaFasterRcnn
from MRCNN.utils import LossWeight



config = Config(GPUS=0,
                NUM_CLASSES=80+1,
                LEARNING_RATE=0.0001,
                TRAIN_IMAGES_PER_GPU=2,
                TEST_IMAGES_PER_GPU=10,
                PRN_IMAGES_PER_GPU=1,
                STEPS_PER_EPOCH=2000,
                VALIDATION_STEPS=50,
                NOVEL_CLASSES=(89,80,14,23),
                SHOTS=0,
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
    model = MetaFasterRcnn(config)
    model.load_weights('save_2023-03-13T03:22:14.336644/chpt/fine_tune/best')


###########################
# phase 1 - FPN+ train
###########################
train_loader = DataLoader(config, Mode.TRAIN, dataset=train_dataset,augmentations=augmentations, phase=1, batch_size=10, prn_batch_size=1)
val_loader = DataLoader(config, Mode.TEST, dataset=val_dataset, phase=1)

lr_schedule = CustomScheduler(config.LEARNING_RATE, 100*config.STEPS_PER_EPOCH,0.1,1, staircase=True)
with config.STRATEGY.scope():
    model.compile(val_dataset, train_loader.active_class_ids)
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=config.GRADIENT_CLIP_NORM)
    model.compile(val_dataset, train_loader.active_class_ids,optimizer=optimizer, train_layers=TrainLayers.FPN_P)

callbacks = [keras.callbacks.ModelCheckpoint(f'save_{now}/chpt/phase1/fpn_p/best',monitor='val_mAP50',save_best_only=True, save_weights_only=True,mode='max'),
            keras.callbacks.TensorBoard(log_dir=f'save_{now}/logs/phase1/fpn_p'),
            keras.callbacks.EarlyStopping('val_mAP50',patience=10,verbose=1, mode='max',restore_best_weights=True)]

model.fit(iter(train_loader), 
        epochs=300000,
        callbacks=callbacks,
        validation_data=iter(val_loader), 
        steps_per_epoch=config.STEPS_PER_EPOCH,
        validation_steps=config.VALIDATION_STEPS)


###########################
# phase 1 - finetune
###########################
train_loader = DataLoader(config, Mode.TRAIN, dataset=train_dataset,augmentations=augmentations, phase=1)
val_loader = DataLoader(config, Mode.TEST, dataset=val_dataset, phase=1)

lr_schedule = CustomScheduler(config.LEARNING_RATE/10, 100*config.STEPS_PER_EPOCH,0.1,1, staircase=True)
with config.STRATEGY.scope():
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=config.GRADIENT_CLIP_NORM)
    model.compile(val_dataset, train_loader.active_class_ids,optimizer=optimizer, train_layers=TrainLayers.ALL)

callbacks = [keras.callbacks.ModelCheckpoint(f'save_{now}/chpt/phase1/all/best',monitor='val_mAP50',save_best_only=True, save_weights_only=True,mode='max'),
            keras.callbacks.TensorBoard(log_dir=f'save_{now}/logs/phase1/all'),
            keras.callbacks.EarlyStopping('val_mAP50',patience=10,verbose=1, mode='max',restore_best_weights=True)]

model.fit(iter(train_loader), 
        epochs=300000,
        callbacks=callbacks,
        validation_data=iter(val_loader), 
        steps_per_epoch=config.STEPS_PER_EPOCH,
        validation_steps=config.VALIDATION_STEPS)


###########################
# phase 2
###########################
train_loader = DataLoader(config, Mode.TRAIN, dataset=train_dataset,augmentations=augmentations, phase=2)
val_loader = DataLoader(config, Mode.TEST, dataset=val_dataset, phase=2)

lr_schedule = CustomScheduler(config.LEARNING_RATE/10, 100*config.STEPS_PER_EPOCH,0.1,1, staircase=True)
with config.STRATEGY.scope():
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=config.GRADIENT_CLIP_NORM)
    model.compile(val_dataset, train_loader.active_class_ids,optimizer=optimizer)

callbacks = [keras.callbacks.ModelCheckpoint(f'save_{now}/chpt/phase2/all/best',monitor='val_mAP50',save_best_only=True, save_weights_only=True,mode='max'),
            keras.callbacks.TensorBoard(log_dir=f'save_{now}/logs/phase2/all'),
            keras.callbacks.EarlyStopping('val_mAP50',patience=10,verbose=1, mode='max',restore_best_weights=True)]

model.fit(iter(train_loader), 
        epochs=300000,
        callbacks=callbacks,
        validation_data=iter(val_loader), 
        steps_per_epoch=config.STEPS_PER_EPOCH,
        validation_steps=config.VALIDATION_STEPS)


result = model.evaluate(iter(val_loader),steps=500)
print(result)
