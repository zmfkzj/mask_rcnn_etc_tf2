import datetime
import imgaug.augmenters as iaa
import keras.api._v2.keras as keras
import tensorflow as tf
import os

from MRCNN.config import Config
from MRCNN.data.data_loader import DataLoader, Mode
from MRCNN.data.dataset import Dataset
from MRCNN.model.mask_rcnn import EvalType, LossWeight, MaskRcnn, TrainLayers



class TrainConfig(Config):
    # GPUS = 0,1
    GPUS = 0
    NUM_CLASSES = 1+80 
    LEARNING_RATE = 0.001
    TRAIN_IMAGES_PER_GPU = 3
    TEST_IMAGES_PER_GPU = 8
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 20
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

lr_schedule = CustomScheduler(config.LEARNING_RATE, 5*config.STEPS_PER_EPOCH,0.9,1, staircase=True)

augmentations = iaa.Sequential([
    iaa.Fliplr(0.5),
    # iaa.GaussianBlur(),
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



with config.STRATEGY.scope():
    model = MaskRcnn(config)
    # model.load_weights('save_2023-03-05T21:04:55.322619/chpt/0/rpn/best')
    # model.compile(val_dataset,EvalType.SEGM, active_class_ids)


count = 0
while True:

    with config.STRATEGY.scope():
        optimizer = keras.optimizers.Nadam(learning_rate=0.00001, clipnorm=config.GRADIENT_CLIP_NORM)
        model.compile(val_dataset,EvalType.SEGM, active_class_ids,optimizer=optimizer,train_layers=TrainLayers.RPN)

    callbacks = [keras.callbacks.ModelCheckpoint(f'save_{now}/chpt/{count}/rpn/best',monitor='rpn_class_loss',save_best_only=True, save_weights_only=True,mode='min'),
                keras.callbacks.TensorBoard(log_dir=f'save_{now}/logs/{count}/rpn_train/',),
                keras.callbacks.EarlyStopping('rpn_class_loss',patience=5,verbose=1, mode='min')]

    train_loader = DataLoader(config, Mode.TRAIN, 12*config.GPU_COUNT, active_class_ids=active_class_ids, dataset=train_dataset,augmentations=augmentations)
    val_loader = DataLoader(config, Mode.TEST,20*config.GPU_COUNT, active_class_ids=active_class_ids, dataset=val_dataset)
    hist = model.fit(iter(train_loader), 
            epochs=300000,
            callbacks=callbacks,
            validation_data=iter(val_loader), 
            steps_per_epoch=500,
            validation_steps=40)


    callbacks = [keras.callbacks.ModelCheckpoint(f'save_{now}/chpt/{count}/fine_tune/best',monitor='val_mAP50',save_best_only=True, save_weights_only=True,mode='max'),
                keras.callbacks.TensorBoard(log_dir=f'save_{now}/logs/{count}/fine_tune'),
                keras.callbacks.EarlyStopping('val_mAP50',patience=5,verbose=1, mode='max')]

    with config.STRATEGY.scope():
        optimizer = keras.optimizers.Nadam(learning_rate=0.00001, clipnorm=config.GRADIENT_CLIP_NORM)
        model.compile(val_dataset,EvalType.SEGM, active_class_ids,optimizer=optimizer,train_layers=TrainLayers.EXC_RPN)

    train_loader = DataLoader(config, Mode.TRAIN, 3*config.GPU_COUNT, active_class_ids=active_class_ids, dataset=train_dataset,augmentations=augmentations)
    val_loader = DataLoader(config, Mode.TEST,20*config.GPU_COUNT, active_class_ids=active_class_ids, dataset=val_dataset)
    hist = model.fit(iter(train_loader), 
            epochs=300000,
            callbacks=callbacks,
            validation_data=iter(val_loader), 
            steps_per_epoch=config.STEPS_PER_EPOCH,
            validation_steps=40)

    count +=1


result = model.evaluate(iter(val_loader),steps=1000)
print(result)
