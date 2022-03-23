import numpy as np
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
    LEARNING_RATE = 0.0001
    IMAGES_PER_GPU = 3
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 100
    

train_config = TrainConfig()
class CustomScheduler(keras.optimizers.schedules.ExponentialDecay):
    def __init__(self, initial_learning_rate, decay_steps, decay_rate, burnin_step, staircase=False, name=None):
        super().__init__(initial_learning_rate, decay_steps, decay_rate, staircase, name)
        self.burnin_step = burnin_step
        # self.initial_learning_rate = initial_learning_rate/train_config.STEPS_PER_EPOCH/train_config.GPU_COUNT
        self.initial_learning_rate = initial_learning_rate/train_config.GPU_COUNT

    def __call__(self, step):
        super_lr = super().__call__(step)
        return tf.cond(step<=self.burnin_step, 
                lambda : self.initial_learning_rate*tf.math.pow(step/self.burnin_step,4),
                lambda : super_lr)


train_dataset = CocoDataset()
train_dataset.load_coco('c:/coco/train2017/', 'c:/coco/annotations/instances_train2017.json')

model = MaskRCNN(train_config).load_weights('mask_rcnn_coco.h5')

lr_schedule = CustomScheduler(train_config.LEARNING_RATE, 80000,0.1,1000, staircase=True)

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
# optimizer = keras.optimizers.SGD(learning_rate=lr_schedule, momentum=train_config.LEARNING_MOMENTUM)


# val_evaluator = Evaluator(model, 'C:/coco/train2017/', 'C:/coco/annotations/instances_train2017.json',train_config, conf_thresh=0.25, iouType='bbox')
val_evaluator = Evaluator(model, 'C:/coco/val2017/', 'C:/coco/annotations/instances_val2017.json',train_config, conf_thresh=0.25, iouType='bbox')

augmentation = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(),
    iaa.Add(per_channel=True),
    iaa.Multiply(per_channel=True),
    iaa.GammaContrast(per_channel=True)

])

trainer = Trainer(model, train_dataset, config=train_config, optimizer=optimizer, val_evaluator=val_evaluator)
trainer.train(40, 'heads')
trainer.train(90, '4+')
trainer.train(160, 'all')

class ValConfig(Config):
    GPUS = 0,1
    # GPUS = 0
    NUM_CLASSES = 1+80 
    IMAGES_PER_GPU = 20

model = MaskRCNN(ValConfig())
model.load_weights('save_model/ckpt-65')

val_evaluator = Evaluator(model, 'C:/coco/val2017/', 'C:/coco/annotations/instances_val2017.json',ValConfig(), conf_thresh=0.25, iouType='bbox')
metric = val_evaluator.eval(save_dir='d:/')
print(metric)

# classes = [info['name'] for info in train_dataset.class_info]
# detector = Detector(model, classes, config)
# detections = detector.detect('d:/coco/val2017/', shuffle=True, limit_step=10)
