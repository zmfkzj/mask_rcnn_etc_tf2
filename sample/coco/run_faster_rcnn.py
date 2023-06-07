import datetime
import numpy as np
import tensorflow as tf
import os

from MRCNN.config import Config
from MRCNN.data.frcnn_data_loader import *
from MRCNN.data.dataset import Dataset
from MRCNN.model import FasterRcnn
from MRCNN.enums import TrainLayers, Mode
import sys
sys.setrecursionlimit(10**6)

# tf.config.run_functions_eagerly(True)

policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)


config = Config(GPUS=0,
                LEARNING_RATE=0.0001,
                TRAIN_IMAGES_PER_GPU=5,
                TEST_IMAGES_PER_GPU=10,
                )


now = datetime.datetime.now().isoformat()
train_dataset = Dataset.from_json('/home/jovyan/dataset/coco/annotations/instances_train2017.json', 
                                  '/home/jovyan/dataset/coco/train2017/')
val_dataset = Dataset.from_json('/home/jovyan/dataset/coco/annotations/instances_val2017.json', 
                                '/home/jovyan/dataset/coco/val2017/')

if not os.path.isdir(f'save_{now}/chpt'):
    os.makedirs(f'save_{now}/chpt')

with config.STRATEGY.scope():
    model = FasterRcnn(config, train_dataset)


train_loader = make_train_dataloader(train_dataset, config)
val_loader = make_test_dataloader(val_dataset, config)

callbacks = [tf.keras.callbacks.ModelCheckpoint(f'save_{now}/chpt/fpn_p/best.h5',monitor='val_mAP85',save_best_only=True, save_weights_only=True,mode='max'),
            tf.keras.callbacks.TensorBoard(log_dir=f'save_{now}/logs/fpn_p'),
            tf.keras.callbacks.ReduceLROnPlateau(),
            tf.keras.callbacks.EarlyStopping('val_mAP50',patience=10,verbose=1, mode='max',restore_best_weights=True)]


###########################
# FPN+ train
###########################
model.set_trainable(TrainLayers.FPN_P)

with config.STRATEGY.scope():
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE, amsgrad=True)
    model.compile(optimizer=optimizer)


model.fit(train_loader, 
        epochs=10000,
        callbacks=callbacks,
        validation_data=val_loader, 
        steps_per_epoch=config.STEPS_PER_EPOCH,
        validation_steps=config.VALIDATION_STEPS)


###########################
# finetune
###########################
model.set_trainable(TrainLayers.ALL)

with config.STRATEGY.scope():
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE, amsgrad=True)
    model.compile(optimizer=optimizer)


model.fit(train_loader, 
        epochs=10000,
        callbacks=callbacks,
        validation_data=val_loader, 
        steps_per_epoch=config.STEPS_PER_EPOCH,
        validation_steps=config.VALIDATION_STEPS)


###########################
# evaluate
###########################
result = model.evaluate(val_loader,steps=1000)
print(result)
