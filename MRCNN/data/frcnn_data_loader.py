import os
from functools import partial
from pathlib import Path

import tensorflow as tf

from MRCNN.config import Config

from .dataset import Dataset
from .utils import *


def make_predict_dataloader(image_pathes:str|list[str], config: Config):
    batch_size = config.TEST_BATCH_SIZE

    if isinstance(image_pathes, str):
        image_pathes = [image_pathes]
    
    pathes = []
    image_ext = ['.jpg', '.jpeg','.png', '.bmp']
    for p in image_pathes:
        if os.path.isdir(p):
            for r,_,fs in os.walk(p):
                for f in fs:
                    if Path(f).suffix.lower() in image_ext:
                        full_path = Path(r)/f
                        pathes.append(str(full_path))
        elif os.path.isfile(p):
            if Path(p).suffix.lower() in image_ext:
                pathes.append(p)
        else:
            raise FileNotFoundError(f'{p}를 찾을 수 없습니다.')
    
    preprocessing = partial(preprocessing_predict,
                            resize_shape=tuple( config.IMAGE_SHAPE[:2] ), 
                            pixel_mean=config.PIXEL_MEAN, 
                            pixel_std=config.PIXEL_STD)
        
    data_loader = tf.data.Dataset\
        .from_tensor_slices(pathes)\
        .map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE)\
        .batch(batch_size)\
        .prefetch(tf.data.AUTOTUNE)
    return data_loader


def make_test_dataloader(dataset: Dataset, config: Config, batch_size=None):
    batch_size = batch_size or config.TEST_BATCH_SIZE

    pathes, img_ids = zip(*[[ img.path, img.id ] for img in dataset.images])
    pathes = tf.data.Dataset.from_tensor_slices(list(pathes))
    img_ids = tf.data.Dataset.from_tensor_slices(list(img_ids))

    preprocessing = partial(preprocessing_test,
                            resize_shape=tuple( config.IMAGE_SHAPE[:2] ), 
                            pixel_mean=config.PIXEL_MEAN, 
                            pixel_std=config.PIXEL_STD)

    data_loader = tf.data.Dataset\
        .zip((pathes,img_ids))\
        .map(lambda pathes, img_ids: preprocessing(path=pathes, img_id=img_ids), num_parallel_calls=tf.data.AUTOTUNE)\
        .batch(batch_size)\
        .prefetch(tf.data.AUTOTUNE)
    return data_loader


def make_train_dataloader(dataset:Dataset, config: Config, batch_size=None):
    batch_size = batch_size or config.TRAIN_BATCH_SIZE

    anchors = get_anchors(config)

    pathes = tf.data.Dataset\
        .from_tensor_slices([img.path for img in dataset.images])

    ann_ids = [[ann.id for ann in img.annotations] for img in dataset.images]
    ann_ids = [ padding_ann_ids(ids, config.MAX_GT_INSTANCES) for ids in ann_ids]
    ann_ids = tf.data.Dataset.from_tensor_slices(ann_ids)

    augmentor = get_augmentor(config)

    preprocessing = partial(preprocessing_train,
                            resize_shape=tuple( config.IMAGE_SHAPE[:2] ), 
                            pixel_mean=config.PIXEL_MEAN, 
                            pixel_std=config.PIXEL_STD, 
                            anchors=anchors, 
                            rpn_train_anchors_per_image=config.RPN_TRAIN_ANCHORS_PER_IMAGE,
                            rpn_bbox_std_dev=config.RPN_BBOX_STD_DEV,
                            max_gt_instances=config.MAX_GT_INSTANCES, 
                            mini_mask_shape=config.MINI_MASK_SHAPE,
                            dataset=dataset,
                            augmentor=augmentor)

    data_loader = tf.data.Dataset\
        .zip((pathes, ann_ids))\
        .repeat()\
        .shuffle(len(dataset.images))\
        .map(lambda pathes, ann_ids: preprocessing(path=pathes, ann_ids=ann_ids), num_parallel_calls=tf.data.AUTOTUNE)\
        .batch(batch_size)\
        .prefetch(tf.data.AUTOTUNE)
    return data_loader
