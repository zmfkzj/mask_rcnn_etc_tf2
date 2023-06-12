import datetime
import numpy as np
import tensorflow as tf
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel(logging.ERROR)

from MRCNN.config import Config
from MRCNN.data.frcnn_data_loader import *
from MRCNN.data.dataset import Dataset
from MRCNN.model import FasterRcnn
from MRCNN.enums import TrainLayers
from MRCNN.model.backbones import backbones

import sys
sys.setrecursionlimit(10**6)

policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)


now = datetime.datetime.now().isoformat()
train_dataset = Dataset.from_json('/home/jovyan/dataset/detection_comp/train.json', 
                                '/home/jovyan/dataset/detection_comp/train')
val_dataset = Dataset.from_json('/home/jovyan/dataset/detection_comp/val.json', 
                                '/home/jovyan/dataset/detection_comp/train')


for b in backbones:
    config = Config(GPUS=0,
                    LEARNING_RATE=0.00000001,
                    TRAIN_IMAGES_PER_GPU=8,
                    TEST_IMAGES_PER_GPU=10,
                    BACKBONE=b,
                    FPN='NASFPN',
                    # FPN='FPN',
                    STEPS_PER_EPOCH=1000,
                    VALIDATION_STEPS=200
                    )

    if not os.path.isdir(f'save_{b}_{now}/chpt'):
        os.makedirs(f'save_{b}_{now}/chpt')

    with config.STRATEGY.scope():
        model = FasterRcnn(config, train_dataset)

    # if b == 'convnext_base':
    #     model.load_weights('convnext_base_save_2023-06-11T07:13:54.093074/chpt/head/best')
    #     model.compile()


    ###########################
    # Head train
    ###########################
    train_loader = make_train_dataloader(train_dataset, config, 20)
    val_loader = make_test_dataloader(val_dataset, config)

    callbacks = [tf.keras.callbacks.ModelCheckpoint(f'save_{b}_{now}/chpt/head/best',monitor='val_mAP85',save_best_only=True, save_weights_only=True,mode='max'),
                tf.keras.callbacks.TensorBoard(log_dir=f'save_{b}_{now}/logs/head')]

    model.set_trainable(TrainLayers.FPN_P)

    with config.STRATEGY.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE, amsgrad=True)
        # optimizer = tf.keras.optimizers.SGD(learning_rate=config.LEARNING_RATE, momentum=config.LEARNING_MOMENTUM, weight_decay=config.WEIGHT_DECAY, clipnorm=config.GRADIENT_CLIP_NORM)
        model.compile(optimizer=optimizer)


    model.fit(train_loader, 
            epochs=1,
            callbacks=callbacks,
            validation_data=val_loader, 
            steps_per_epoch=config.STEPS_PER_EPOCH,
            validation_steps=config.VALIDATION_STEPS)


    ###########################
    # finetune
    ###########################
    train_loader = make_train_dataloader(train_dataset, config)
    val_loader = make_test_dataloader(val_dataset, config)

    callbacks = [tf.keras.callbacks.ModelCheckpoint(f'save_{b}_{now}/chpt/fingtune/best',monitor='val_mAP85',save_best_only=True, save_weights_only=True,mode='max'),
                tf.keras.callbacks.TensorBoard(log_dir=f'save_{b}_{now}/logs/fingtune'),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mAP85', mode='max'),
                tf.keras.callbacks.EarlyStopping('val_mAP85',patience=10,verbose=1, mode='max',restore_best_weights=True)]

    model.set_trainable(TrainLayers.ALL)

    with config.STRATEGY.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE/10, amsgrad=True)
        # optimizer = tf.keras.optimizers.SGD(learning_rate=config.LEARNING_RATE/10, momentum=config.LEARNING_MOMENTUM, weight_decay=config.WEIGHT_DECAY, clipnorm=config.GRADIENT_CLIP_NORM)
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
