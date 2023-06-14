import datetime
import numpy as np
import os

from MRCNN.layer.fpn import FPN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import logging
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
    if not os.path.isdir(f'save_{b}_{now}/chpt'):
        os.makedirs(f'save_{b}_{now}/chpt')

    config = Config(GPUS=0,
                    LEARNING_RATE=0.0001,
                    TRAIN_IMAGES_PER_GPU=10,
                    TEST_IMAGES_PER_GPU=10,
                    BACKBONE=b,
                    FPN='NASFPN',
                    STEPS_PER_EPOCH=300,
                    VALIDATION_STEPS=300,
                    RPN_NMS_THRESHOLD=0.5,
                    IMAGE_SHAPE=np.array([512,512,3])
                    )

    with config.STRATEGY.scope():
        model = FasterRcnn(config, val_dataset)

    # model.load_weights('save_ResNet101_2023-06-13T12:45:22.885073/chpt/fingtune/train_loss')
    # model.compile()


    ###########################
    # NAS FPN train
    ###########################

    train_loader = make_train_dataloader(train_dataset, config)
    val_loader = make_test_dataloader(val_dataset, config)

    callbacks = [tf.keras.callbacks.ModelCheckpoint(f'save_{b}_{now}/chpt/nas_fpn/best',monitor='val_mAP85',save_best_only=True, save_weights_only=True,mode='max'),
                tf.keras.callbacks.ModelCheckpoint(f'save_{b}_{now}/chpt/nas_fpn/train_loss',monitor='mean_loss',save_best_only=True, save_weights_only=True,mode='min'),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='mean_loss', mode='min',patience=5),
                tf.keras.callbacks.TensorBoard(log_dir=f'save_{b}_{now}/logs/nas_fpn')]

    model.neck.nas_train = True
    with config.STRATEGY.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE, amsgrad=True)
        # optimizer = tf.keras.optimizers.SGD(learning_rate=config.LEARNING_RATE, momentum=config.LEARNING_MOMENTUM, weight_decay=config.WEIGHT_DECAY, clipnorm=config.GRADIENT_CLIP_NORM)
        model.compile(optimizer=optimizer)


    model.fit(train_loader, 
            epochs=10000,
            callbacks=callbacks,
            validation_data=val_loader, 
            steps_per_epoch=config.STEPS_PER_EPOCH,
            validation_steps=config.VALIDATION_STEPS)

    model.neck.nas_train = False


    ###########################
    # Head train
    ###########################
    train_loader = make_train_dataloader(train_dataset, config)
    val_loader = make_test_dataloader(val_dataset, config)

    callbacks = [tf.keras.callbacks.ModelCheckpoint(f'save_{b}_{now}/chpt/head/best',monitor='val_mAP85',save_best_only=True, save_weights_only=True,mode='max'),
                tf.keras.callbacks.ModelCheckpoint(f'save_{b}_{now}/chpt/head/train_loss',monitor='mean_loss',save_best_only=True, save_weights_only=True,mode='min'),
                tf.keras.callbacks.TensorBoard(log_dir=f'save_{b}_{now}/logs/head')]

    model.set_trainable(TrainLayers.FPN_P)

    with config.STRATEGY.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE, amsgrad=True)
        # optimizer = tf.keras.optimizers.SGD(learning_rate=config.LEARNING_RATE, momentum=config.LEARNING_MOMENTUM, weight_decay=config.WEIGHT_DECAY, clipnorm=config.GRADIENT_CLIP_NORM)
        model.compile(optimizer=optimizer)


    model.fit(train_loader, 
            epochs=2,
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
                tf.keras.callbacks.ModelCheckpoint(f'save_{b}_{now}/chpt/fingtune/train_loss',monitor='mean_loss',save_best_only=True, save_weights_only=True,mode='min'),
                tf.keras.callbacks.TensorBoard(log_dir=f'save_{b}_{now}/logs/fingtune'),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='mean_loss', mode='min',patience=5),
                tf.keras.callbacks.EarlyStopping('val_mAP85',patience=5,verbose=1, mode='max',restore_best_weights=True)]

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
    result = model.evaluate(val_loader,steps=400)
    for metric, value in result.items():
        print(metric, value.numpy())
