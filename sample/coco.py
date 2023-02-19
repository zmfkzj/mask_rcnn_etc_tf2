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

# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)


class TrainConfig(Config):
    GPUS = 0,1
    # GPUS = 0
    NUM_CLASSES = 1+80 
    LEARNING_RATE = 0.0001
    IMAGES_PER_GPU = 8
    STEPS_PER_EPOCH = 3000
    VALIDATION_STEPS = 200
    

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

lr_schedule = CustomScheduler(config.LEARNING_RATE, 9000,0.95,1000, staircase=True)

augmentations = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(),
    # iaa.Add(per_channel=True),
    # iaa.Multiply(per_channel=True),
    # iaa.GammaContrast(per_channel=True)
])

now = datetime.datetime.now().isoformat()
train_dataset = Dataset('/home/tmdocker/host/dataset/coco/coco/annotations/instances_train2017.json', 
                    '/home/tmdocker/host/dataset/coco/coco/train2017/')
val_dataset = Dataset('/home/tmdocker/host/dataset/coco/coco/annotations/instances_val2017.json', 
                    '/home/tmdocker/host/dataset/coco/coco/val2017/')

active_class_ids = [cat['id'] for cat in train_dataset.coco.dataset['categories']]

train_loader = DataLoader(config, active_class_ids, Mode.TRAIN, config.BATCH_SIZE, dataset=train_dataset,augmentations=augmentations)
val_loader = DataLoader(config, active_class_ids, Mode.TEST,config.TEST_BATCH_SIZE, dataset=val_dataset)

callbacks = [keras.callbacks.ModelCheckpoint(f'save_{now}',monitor='mAP50',save_best_only=True, save_weights_only=True),
             keras.callbacks.TensorBoard(log_dir=f'logs_{now}'),
             keras.callbacks.EarlyStopping('mAP50')]

with config.STRATEGY.scope():
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=config.GRADIENT_CLIP_NORM)
    val_metric = CocoMetric(val_dataset, config, active_class_ids,eval_type=EvalType.SEGM)
    model = MaskRcnn(config,train_dataset)

    model.set_trainable(TrainLayers.HEADS)
    model.compile(val_metric,optimizer=optimizer)

model.fit(iter(train_loader), epochs=100,callbacks=callbacks,validation_data=iter(val_loader), steps_per_epoch=config.STEPS_PER_EPOCH,validation_steps=config.VALIDATION_STEPS)


with config.STRATEGY.scope():
    model.set_trainable(TrainLayers.ALL)
    model.compile(val_metric,optimizer=optimizer)

model.fit(iter(train_loader), epochs=300,callbacks=callbacks,validation_data=iter(val_loader), steps_per_epoch=config.STEPS_PER_EPOCH,validation_steps=config.VALIDATION_STEPS)

model.evaluate(iter(val_loader))