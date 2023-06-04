import os
from dataclasses import dataclass
from pathlib import Path
from albumentations import BaseCompose

import tensorflow as tf

from MRCNN.utils import (compute_backbone_shapes,
                         generate_pyramid_anchors)
from .dataset import Dataset


@dataclass
class DataLoader:
    def __post_init__(self):
        backbone_shapes = compute_backbone_shapes(self.config)
        self.anchors = generate_pyramid_anchors(
                            self.config.RPN_ANCHOR_SCALES,
                            self.config.RPN_ANCHOR_RATIOS,
                            backbone_shapes,
                            self.config.BACKBONE_STRIDES,
                            self.config.RPN_ANCHOR_STRIDE)
        
    @staticmethod
    def make_predict_dataloader(image_pathes:str|list[str], resize_shape:tuple[int,int], batch_size:int=1, augmentor:BaseCompose=None):

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
            
        data_loader = tf.data.Dataset\
            .from_tensor_slices(pathes)\
            .map(lambda p: preprocessing_predict(p), num_parallel_calls=tf.data.AUTOTUNE)\
            .batch(batch_size)\
            .map(lambda datas: 
                    [InputDatas(self.config, self.anchors.shape[0], **datas).to_dict()],
                    num_parallel_calls=tf.data.AUTOTUNE)\
            .prefetch(tf.data.AUTOTUNE)
        return data_loader
    

    def make_test_dataloader(self):
        coco = self.dataset.coco
        self.batch_size = self.config.TEST_BATCH_SIZE if self.batch_size is None else self.batch_size

        self.set_active_class_ids()

        pathes = tf.data.Dataset\
            .from_tensor_slices([img['path'] for img in coco.dataset['images']])
        img_ids = tf.data.Dataset\
            .from_tensor_slices([img['id'] for img in coco.dataset['images']])

        data_loader = tf.data.Dataset\
            .zip((pathes,img_ids))\
            .shuffle(len(self.dataset))\
            .map(self.preprocessing_test, num_parallel_calls=tf.data.AUTOTUNE)\
            .batch(self.batch_size)\
            .map(lambda datas: 
                    [InputDatas(self.config, self.anchors.shape[0], **datas).to_dict()],
                    num_parallel_calls=tf.data.AUTOTUNE)\
            .repeat()\
            .prefetch(tf.data.AUTOTUNE)
        return data_loader


    def make_train_dataloader(self):
        coco = self.dataset.coco
        self.batch_size = self.config.TRAIN_BATCH_SIZE if self.batch_size is None else self.batch_size

        self.set_active_class_ids()

        path = tf.data.Dataset\
            .from_tensor_slices([img['path'] for img in coco.dataset['images']])

        ann_ids = [self.padding_ann_ids(coco.getAnnIds(img['id'], self.active_class_ids)) 
                    for img in coco.dataset['images']]
        ann_ids = tf.data.Dataset.from_tensor_slices(ann_ids)

        data_loader = tf.data.Dataset\
            .zip((path, ann_ids))\
            .repeat()\
            .shuffle(len(self.dataset))\
            .map(lambda path, ann_ids: self.preprocessing_train(path, ann_ids), 
                    num_parallel_calls=tf.data.AUTOTUNE)\

        active_classes = [1]+[1 if cat in self.active_class_ids else 0 
                                for cat in self.active_class_ids]

        active_classes_dataset = tf.data.Dataset\
            .from_tensors(active_classes)\
            .repeat()

        data_loader = tf.data.Dataset\
            .zip((data_loader, active_classes_dataset))\
            .batch(self.batch_size)\
            .map(lambda datas, active_class_id: 
                 [InputDatas(self.config, 
                             self.anchors.shape[0], 
                             **datas).update_active_class_ids(active_class_id).to_dict()])\
            .prefetch(tf.data.AUTOTUNE)
        return data_loader


@tf.function
def preprocessing_predict(path):
    image = self.load_image(path)
    resized_image, window = self.resize_image(image, list(self.config.IMAGE_SHAPE[:2]))
    preprocessed_image = self.config.PREPROCESSING(tf.cast(resized_image, tf.float32))
    origin_image_shape = tf.shape(image)
    return {'input_images': preprocessed_image,
            'input_window': window,
            'origin_image_shapes':origin_image_shape,
            'pathes':path}

# @tf.function
def preprocessing_test(self, path, img_id):
    image = self.load_image(path)
    resized_image, window = self.resize_image(image, list(self.config.IMAGE_SHAPE[:2]))
    preprocessed_image = self.config.PREPROCESSING(tf.cast(resized_image, tf.float32))
    origin_image_shape = tf.shape(image)
    return {'input_images': preprocessed_image,
            'input_window': window,
            'origin_image_shapes':origin_image_shape,
            'image_ids':img_id}

@tf.function
def preprocessing_train(self, path, ann_ids):
    image = self.load_image(path)
    boxes, dataloader_class_ids =\
        tf.py_function(self.load_gt, (ann_ids,tf.shape(image)[0],tf.shape(image)[1]),(tf.float32, tf.int64))

    if self.augmentations is not None:
        image, boxes = \
            self.augment(image, boxes)

    resized_image, resized_boxes = \
        self.resize(image, boxes)
    
    preprocessed_image = self.config.PREPROCESSING(tf.cast(resized_image, tf.float32))

    rpn_match, rpn_bbox = \
        self.build_rpn_targets(dataloader_class_ids, resized_boxes)

    
    pooled_box = tf.zeros([self.config.MAX_GT_INSTANCES,4],dtype=tf.float32)
    pooled_class_id = tf.zeros([self.config.MAX_GT_INSTANCES],dtype=tf.int64)

    instance_count = tf.shape(boxes)[0]
    if instance_count>self.config.MAX_GT_INSTANCES:
        indices = tf.random.shuffle(tf.range(instance_count))[:self.config.MAX_GT_INSTANCES]
        indices = tf.expand_dims(indices,1)
    else:
        indices = tf.range(instance_count)
        indices = tf.expand_dims(indices,1)

    resized_boxes = tf.tensor_scatter_nd_update(pooled_box, indices, tf.gather(resized_boxes, tf.squeeze(indices,-1)))
    dataloader_class_ids = tf.tensor_scatter_nd_update(pooled_class_id, indices, tf.gather(dataloader_class_ids, tf.squeeze(indices, -1)))

    return {'input_images': preprocessed_image,
            'input_gt_boxes': resized_boxes,
            'dataloader_class_ids':dataloader_class_ids,
            'rpn_match':rpn_match,
            'rpn_bbox':rpn_bbox}

