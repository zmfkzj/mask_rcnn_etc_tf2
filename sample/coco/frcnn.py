import datetime
import imgaug.augmenters as iaa
import keras.api._v2.keras as keras
import numpy as np
import tensorflow as tf
import os

from MRCNN.config import Config
from MRCNN.data.frcnn_data_loader import DataLoader
from MRCNN.data.dataset import Dataset
from MRCNN.model import MaskRcnn
from MRCNN.enums import EvalType, Model, TrainLayers, Mode
from MRCNN.model.faster_rcnn import FasterRcnn
from MRCNN.utils import LossWeight



class TrainConfig(Config):
    # GPUS = 0,1
    GPUS = 0
    NUM_CLASSES = 1+80 
    LEARNING_RATE = 0.0001
    TRAIN_IMAGES_PER_GPU = 3
    TEST_IMAGES_PER_GPU = 8
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 50
    # MAX_GT_INSTANCES = 25
    # RPN_TRAIN_ANCHORS_PER_IMAGE = 1024
    # BACKBONE = staticmethod(keras.applications.ResNet152V2)
    # PREPROCESSING = staticmethod(keras.applications.resnet_v2.preprocess_input)
    # TOP_DOWN_PYRAMID_SIZE = 512
    # RPN_ANCHOR_SCALES =((16,32), (64,96), (128,192), (256,384), (512,768))
    

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

lr_schedule = CustomScheduler(config.LEARNING_RATE, 10*config.STEPS_PER_EPOCH,0.9,1, staircase=True)

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

if not os.path.isdir(f'save_{now}/chpt'):
    os.makedirs(f'save_{now}/chpt')



# with config.STRATEGY.scope():
#     model = FasterRcnn(config)
#     model.load_weights('save_2023-03-07T10:47:43.225611/chpt/rpn/best')
#     model.compile(val_dataset, active_class_ids)
    
#     optimizer = keras.optimizers.Nadam(learning_rate=0.001, clipnorm=config.GRADIENT_CLIP_NORM)
#     model.compile(val_dataset, active_class_ids,optimizer=optimizer)

# callbacks = [keras.callbacks.ModelCheckpoint(f'save_{now}/chpt/rpn/best',monitor='val_rpn_class_loss',save_best_only=True, save_weights_only=True,mode='min'),
#             keras.callbacks.TensorBoard(log_dir=f'save_{now}/logs/rpn_train/',),
#             keras.callbacks.EarlyStopping('val_rpn_class_loss',patience=5,verbose=1, mode='min')]

# train_loader = DataLoader(config, Mode.TRAIN, 4*config.GPU_COUNT, active_class_ids=active_class_ids, dataset=train_dataset,augmentations=augmentations)
# val_loader = DataLoader(config, Mode.TRAIN,20*config.GPU_COUNT, active_class_ids=active_class_ids, dataset=val_dataset)
# hist = model.fit(iter(train_loader), 
#         epochs=300000,
#         callbacks=callbacks,
#         validation_data=iter(val_loader), 
#         steps_per_epoch=1000,
#         validation_steps=50)


with config.STRATEGY.scope():
    model = FasterRcnn(config)

    optimizer = keras.optimizers.Nadam(learning_rate=lr_schedule, clipnorm=config.GRADIENT_CLIP_NORM)
    model.compile(val_dataset, active_class_ids,optimizer=optimizer)

callbacks = [keras.callbacks.ModelCheckpoint(f'save_{now}/chpt/fine_tune/best',monitor='val_mAP50',save_best_only=True, save_weights_only=True,mode='max'),
            keras.callbacks.TensorBoard(log_dir=f'save_{now}/logs/fine_tune'),
            keras.callbacks.EarlyStopping('val_mAP50',patience=20,verbose=1, mode='max', start_from_epoch=40)]

train_loader = DataLoader(config, Mode.TRAIN, 4*config.GPU_COUNT, active_class_ids=active_class_ids, dataset=train_dataset,augmentations=augmentations)
val_loader = DataLoader(config, Mode.TEST,20*config.GPU_COUNT, active_class_ids=active_class_ids, dataset=val_dataset)
hist = model.fit(iter(train_loader), 
        epochs=300000,
        callbacks=callbacks,
        validation_data=iter(val_loader), 
        steps_per_epoch=config.STEPS_PER_EPOCH,
        validation_steps=50)



result = model.evaluate(iter(val_loader),steps=500)
print(result)
