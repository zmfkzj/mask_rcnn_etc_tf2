import os
from pydantic.dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from official.vision.ops.iou_similarity import iou

import numpy as np
import pickle
import tensorflow as tf
from imgaug.augmentables.bbs import BoundingBoxesOnImage
from imgaug.augmenters import Sequential

from MRCNN.config import Config
from MRCNN.data import frcnn_data_loader
from MRCNN.enums import Mode
from MRCNN.data.dataset import Dataset
from MRCNN.utils import (compute_backbone_shapes,
                         generate_pyramid_anchors)
from pycocotools import mask as maskUtils

class InputDatas(tf.experimental.ExtensionType):
    input_gt_boxes: tf.Tensor
    dataloader_class_ids: tf.Tensor
    rpn_match: tf.Tensor
    rpn_bbox: tf.Tensor
    active_class_ids: tf.Tensor
    prn_images: tf.Tensor
    pathes: tf.Tensor
    input_images: tf.Tensor
    input_window: tf.Tensor
    origin_image_shapes: tf.Tensor
    image_ids: tf.Tensor
    attentions: tf.Tensor

    def replace_data(self, new_dict):
        args = {
            'input_gt_boxes': self.input_gt_boxes,
            'dataloader_class_ids': self.dataloader_class_ids,
            'rpn_match': self.rpn_match,
            'rpn_bbox': self.rpn_bbox,
            'active_class_ids': self.active_class_ids,
            'prn_images': self.prn_images,
            'pathes': self.pathes,
            'input_images': self.input_images,
            'input_window': self.input_window,
            'origin_image_shapes': self.origin_image_shapes,
            'image_ids': self.image_ids,
            'attentions': self.attentions
            }
        args.update(new_dict)
        return InputDatas(**args)


@dataclass
class DataLoader(frcnn_data_loader.DataLoader):

    phase:int=1
    attentions:Optional[str] = None
    batch_size:Optional[int] = None
    prn_batch_size:Optional[int] = None

    def __hash__(self) -> int:
        return hash((tuple(self.config.ACTIVE_CLASS_IDS) if self.config.ACTIVE_CLASS_IDS is not None else None, 
                     self.mode, 
                     tuple(self.image_pathes) if self.image_pathes is not None else None, 
                     self.dataset,
                     tuple(self.novel_classes),
                     self.phase))

    def __post_init__(self):
        backbone_shapes = compute_backbone_shapes(self.config)
        self.anchors = generate_pyramid_anchors(
                            self.config.RPN_ANCHOR_SCALES,
                            self.config.RPN_ANCHOR_RATIOS,
                            backbone_shapes,
                            self.config.BACKBONE_STRIDES,
                            self.config.RPN_ANCHOR_STRIDE)

        default_datas = InputDatas(
            input_gt_boxes = tf.zeros([self.config.TRAIN_BATCH_SIZE,self.config.MAX_GT_INSTANCES,4],dtype=tf.float32),
            dataloader_class_ids = tf.zeros([self.config.TRAIN_BATCH_SIZE,self.config.MAX_GT_INSTANCES],dtype=tf.int64),
            rpn_match = tf.zeros([self.config.TRAIN_BATCH_SIZE,self.anchors.shape[0],1], dtype=tf.int64),
            rpn_bbox = tf.zeros([self.config.TRAIN_BATCH_SIZE,self.config.RPN_TRAIN_ANCHORS_PER_IMAGE,4], dtype=tf.float32),
            active_class_ids = tf.zeros([self.config.TRAIN_BATCH_SIZE,self.config.NUM_CLASSES], dtype=tf.int32),
            prn_images = tf.zeros([self.config.PRN_BATCH_SIZE,self.config.NUM_CLASSES-1,*self.config.PRN_IMAGE_SIZE, 4], dtype=tf.float32),
            pathes = tf.zeros([self.config.TRAIN_BATCH_SIZE],dtype=tf.string),
            input_images = tf.zeros([self.config.TRAIN_BATCH_SIZE, *self.config.IMAGE_SHAPE], dtype=tf.float32),
            input_window = tf.zeros([self.config.TRAIN_BATCH_SIZE,4], dtype=tf.float32),
            origin_image_shapes = tf.zeros([self.config.TRAIN_BATCH_SIZE,2], dtype=tf.int32),
            image_ids = tf.zeros([self.config.TRAIN_BATCH_SIZE], dtype=tf.int32),
            attentions = tf.zeros([self.config.NUM_CLASSES,self.config.FPN_CLASSIF_FC_LAYERS_SIZE], dtype=tf.float32)
        )

        if self.phase==1:
            self.dataset.set_dataloader_class_list(self.config.NOVEL_CLASSES)
        
        self.novel_classes = tuple(self.config.NOVEL_CLASSES)
        if self.config.SHOTS > self.dataset.min_class_count:
            ValueError('SHOTS must be less than min_class_count')
        
        if self.mode in [Mode.TRAIN, Mode.TEST]:
            coco = self.dataset.coco

        if self.mode == Mode.TRAIN:
            self.batch_size = self.config.TRAIN_BATCH_SIZE if self.batch_size is None else self.batch_size
            self.prn_batch_size = self.config.PRN_BATCH_SIZE if self.prn_batch_size is None else self.prn_batch_size

            if self.config.ACTIVE_CLASS_IDS is None:
                if self.phase == 1:
                    self.active_class_ids = tuple([cat for cat in self.dataset.coco.cats if  cat not in self.config.NOVEL_CLASSES])
                else:
                    self.active_class_ids = tuple([cat for cat in self.dataset.coco.cats])
            else:
                if self.phase == 1:
                    self.active_class_ids = tuple([id for id in self.config.ACTIVE_CLASS_IDS if id not in self.config.NOVEL_CLASSES])
                else:
                    self.active_class_ids = tuple(self.config.ACTIVE_CLASS_IDS)

            path = tf.data.Dataset\
                .from_tensor_slices([img['path'] for img in coco.dataset['images']])

            ann_ids = [self.padding_ann_ids(coco.getAnnIds(img['id'], self.active_class_ids)) 
                       for img in coco.dataset['images']]
            ann_ids = tf.data.Dataset.from_tensor_slices(ann_ids)

            self.data_loader = tf.data.Dataset\
                .zip((path, ann_ids))\
                .shuffle(len(self.dataset))\
                .map(lambda path, ann_ids: self.preproccessing_train(path, ann_ids), 
                     num_parallel_calls=tf.data.AUTOTUNE)\


            active_classes = [1]+[1 if cat in self.active_class_ids else 0 
                                  for cat in self.active_class_ids]
            
            active_classes_dataset = tf.data.Dataset\
                .from_tensors(active_classes)\
                .repeat()

            prn_datasets = []
            for dataset_class_id in self.dataset.coco.cats:
                if self.phase==1 and dataset_class_id in self.novel_classes:
                    continue
                else:
                    ann_ids = self.dataset.coco.getAnnIds(catIds=dataset_class_id, iscrowd=False)
                    dataset = tf.data.Dataset\
                        .from_tensor_slices(ann_ids)\
                        .shuffle(len(ann_ids))\
                        .map(self.preprocessing_prn,num_parallel_calls=tf.data.AUTOTUNE)\
                        .filter(lambda prn_img: not tf.reduce_all(tf.math.is_nan(prn_img)))

                    if self.config.SHOTS==0:
                        dataset = dataset.repeat()
                    else:
                        dataset = dataset.take(self.config.SHOTS).repeat()

                    prn_datasets.append(dataset)
            
            prn_dataset = tf.data.Dataset\
                .zip(tuple(prn_datasets))\
                .map(lambda *prn_images: tf.stack(prn_images, 0), num_parallel_calls=tf.data.AUTOTUNE)\
                .batch(self.prn_batch_size)
            

            self.data_loader = tf.data.Dataset\
                .zip((self.data_loader, active_classes_dataset))\
                .map(lambda datas, active_class_id: [*datas, active_class_id],
                     num_parallel_calls=tf.data.AUTOTUNE)\
                .repeat()\
                .batch(self.batch_size)\
            
            self.data_loader = tf.data.Dataset\
                .zip((self.data_loader, prn_dataset))\
                .map(lambda datas, prn_images: [*datas, prn_images],
                     num_parallel_calls=tf.data.AUTOTUNE)\
                .map(lambda resized_image, resized_boxes, dataloader_class_ids,rpn_match, rpn_bbox, active_class_ids, prn_images: 
                     [default_datas.replace_data({
                        'input_gt_boxes' : resized_boxes,
                        'dataloader_class_ids' : dataloader_class_ids,
                        'rpn_match':rpn_match,
                        'rpn_bbox':rpn_bbox,
                        'active_class_ids':active_class_ids,
                        'prn_images':prn_images,
                        'input_images':resized_image,
                        })],
                     num_parallel_calls=tf.data.AUTOTUNE)\
                .prefetch(tf.data.AUTOTUNE)


        elif self.mode == Mode.PREDICT:
            self.batch_size = self.config.TEST_BATCH_SIZE if self.batch_size is None else self.batch_size

            if isinstance(self.image_pathes, str):
                self.image_pathes = [self.image_pathes]

            with open(self.attentions, 'rb') as f:
                attentions = pickle.load(f)

            attentions_dataset = tf.data.Dataset\
                .from_tensors(attentions)\
                .repeat()
            
            pathes = []
            for p in self.image_pathes:
                if os.path.isdir(p):
                    for r,_,fs in os.walk(p):
                        for f in fs:
                            if Path(f).suffix.lower() in ['.jpg', '.jpeg','.png']:
                                full_path = Path(r)/f
                                pathes.append(str(full_path))
                elif os.path.isfile(p):
                    if Path(p).suffix.lower() in ['.jpg', '.jpeg','.png']:
                        pathes.append(p)
                else:
                    raise FileNotFoundError(f'{p}를 찾을 수 없습니다.')
            
                
            self.data_loader = tf.data.Dataset\
                .from_tensor_slices(pathes)\
                .map(lambda p: self.preprocessing_predict(p), num_parallel_calls=tf.data.AUTOTUNE)\
                .batch(self.batch_size)\
            
            self.data_loader = tf.data.Dataset\
                .zip((self.data_loader, attentions_dataset))\
                .map(lambda datas, attentions: [*datas, attentions],
                    num_parallel_calls=tf.data.AUTOTUNE)\
                .map(lambda input_images, input_window, origin_image_shapes, pathes, attentions: 
                     [default_datas.replace_data({
                        'input_images':input_images,
                        'input_window':input_window,
                        'origin_image_shapes' : origin_image_shapes,
                        'pathes' : pathes,
                        'attentions' : attentions
                     })],
                    num_parallel_calls=tf.data.AUTOTUNE)\
                .prefetch(tf.data.AUTOTUNE)
        
        else: # Test
            self.batch_size = self.config.TEST_BATCH_SIZE if self.batch_size is None else self.batch_size

            if self.attentions is not None:
                with open(self.attentions, 'rb') as f:
                    attentions = pickle.load(f)

                attentions_dataset = tf.data.Dataset\
                    .from_tensors(attentions)\
                    .repeat()

            pathes = tf.data.Dataset\
                .from_tensor_slices([img['path'] for img in coco.dataset['images']])
            img_ids = tf.data.Dataset\
                .from_tensor_slices([img['id'] for img in coco.dataset['images']])


            self.data_loader = tf.data.Dataset\
                .zip((pathes,img_ids))\
                .shuffle(len(self.dataset))\
                .map(self.preproccessing_test, num_parallel_calls=tf.data.AUTOTUNE)\
                .batch(self.batch_size)\
            
            if self.attentions is not None:
                self.data_loader = tf.data.Dataset\
                    .zip((self.data_loader, attentions_dataset))\
                    .map(lambda datas, attentions: [*datas, attentions],
                        num_parallel_calls=tf.data.AUTOTUNE)\
                    .map(lambda input_images, input_window, origin_image_shapes, image_ids, attentions : 
                         [default_datas.replace_data({
                            'input_images' : input_images,
                            'input_window':input_window,
                            'origin_image_shapes' : origin_image_shapes,
                            'image_ids' : image_ids,
                            'attentions' : attentions
                         })],
                        num_parallel_calls=tf.data.AUTOTUNE)\
                    .repeat()\
                    .prefetch(tf.data.AUTOTUNE)
            else:
                self.data_loader = self.data_loader\
                    .map(lambda input_images, input_window, origin_image_shapes, image_ids: 
                         [default_datas.replace_data({
                            'input_images' : input_images,
                            'input_window':input_window,
                            'origin_image_shapes' : origin_image_shapes,
                            'image_ids' : image_ids
                         })],
                        num_parallel_calls=tf.data.AUTOTUNE)\
                    .repeat()\
                    .prefetch(tf.data.AUTOTUNE)
            
        with self.config.STRATEGY.scope():
            self.data_loader = self.config.STRATEGY.experimental_distribute_dataset(self.data_loader)


    @tf.function
    def preprocessing_prn(self, ann_id):
        image_id = tf.py_function(lambda id: self.dataset.coco.loadAnns(int(id))[0]['image_id'], (ann_id,),tf.int32)
        path = tf.py_function(lambda id: self.dataset.coco.loadImgs(int(id))[0]['path'], (image_id,),tf.string)
        bbox = tf.py_function(lambda id: tf.constant(self.dataset.coco.loadAnns(int(id))[0]['bbox']), (ann_id,),tf.float32)
        x1 = bbox[0]
        y1 = bbox[1]
        w = bbox[2]
        h = bbox[3]
        bbox = tf.stack((y1,x1,y1+h,x1+w))
        image = self.load_image(path)
        resized_image, resized_bbox = self.resize_prn(image, bbox)
        preprocessed_image = self.config.PREPROCESSING(tf.cast(resized_image, tf.float32))

        mask = tf.py_function(lambda bbox, h, w:self.annToMask(bbox, h, w),(resized_bbox,*self.config.PRN_IMAGE_SIZE),tf.bool)
        if tf.size(tf.where(mask==True))==0:
            return np.nan
        mask = tf.expand_dims(mask, -1)
        mask = tf.cast(mask, tf.float32)
        prn_image = tf.concat([preprocessed_image, mask], -1)
        return prn_image


    @tf.function
    def resize_prn(self, image, bbox):
        origin_shape = tf.shape(image)
        resized_image, window = self.resize_image(image, self.config.PRN_IMAGE_SIZE)
        resized_bbox = self.resize_box(bbox, origin_shape, window)
        return resized_image, resized_bbox
         

    @staticmethod
    def bbox_to_segm(bbox):
        y1,x1,y2,x2 = bbox
        return [[x1,y1,x2,y1,x2,y2,x1,y2]]


    def annToRLE(self, bbox, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = self.bbox_to_segm(bbox)
        rles = maskUtils.frPyObjects(segm, height, width)
        rle = maskUtils.merge(rles)
        return rle

    def annToMask(self, bbox, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(bbox, height, width)
        m = maskUtils.decode(rle)
        return m