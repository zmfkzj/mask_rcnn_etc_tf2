import datetime
import imgaug.augmenters as iaa
import keras.api._v2.keras as keras
import tensorflow as tf
import os

from MRCNN.config import Config
from MRCNN.data.data_loader import DataLoader, Mode
from MRCNN.data.dataset import Dataset
from MRCNN.model.mask_rcnn import EvalType, MaskRcnn, TrainLayers



class TrainConfig(Config):
    GPUS = 0,1
    # GPUS = 0
    NUM_CLASSES = 1+80 
    LEARNING_RATE = 0.0001
    TRAIN_IMAGES_PER_GPU = 3
    TEST_IMAGES_PER_GPU = 8
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 20
    

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

lr_schedule = CustomScheduler(config.LEARNING_RATE, 50000,0.9,1, staircase=True)

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

if not os.path.isdir(f'save_{now}/chpt'):
    os.makedirs(f'save_{now}/chpt')

callbacks = [keras.callbacks.ModelCheckpoint(f'save_{now}/chpt/'+'best',monitor='val_mAP50',save_best_only=True, save_weights_only=True,mode='max'),
             keras.callbacks.TensorBoard(log_dir=f'save_{now}/logs'),
             keras.callbacks.EarlyStopping('val_mAP50',patience=10,start_from_epoch=10)]


with config.STRATEGY.scope():
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=config.GRADIENT_CLIP_NORM)
    model = MaskRcnn(config)

    model.load_weights('save_2023-02-23T10:51:47.314532/chpt/01-0.0000')
    # model.set_trainable(TrainLayers.HEADS)
    model.set_trainable(TrainLayers.ALL)
    model.compile(val_dataset,EvalType.SEGM, active_class_ids,optimizer=optimizer)

model.fit(iter(train_loader), 
          epochs=300000,
          callbacks=callbacks,
          validation_data=iter(val_loader), 
          steps_per_epoch=config.STEPS_PER_EPOCH,
          validation_steps=config.VALIDATION_STEPS)


# with config.STRATEGY.scope():
#     optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=config.GRADIENT_CLIP_NORM)
#     model.set_trainable(TrainLayers.ALL)
#     model.compile(val_dataset,EvalType.SEGM, active_class_ids,optimizer=optimizer)

# model.fit(iter(train_loader), 
#           epochs=300000,
#           callbacks=callbacks,
#           validation_data=iter(val_loader), 
#           steps_per_epoch=config.STEPS_PER_EPOCH,
#           validation_steps=config.VALIDATION_STEPS)

result = model.evaluate(iter(val_loader),steps=1000)
print(result)