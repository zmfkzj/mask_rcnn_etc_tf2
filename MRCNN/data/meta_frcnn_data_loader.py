import os
from pydantic.dataclasses import dataclass
from pydantic import Field
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


@dataclass
class DataLoader(frcnn_data_loader.DataLoader):
    novel_classes:list[int] = ...
    phase:int=1
    attentions:Optional[str] = None

    def __hash__(self) -> int:
        return hash((tuple(self.active_class_ids) if self.active_class_ids is not None else None, 
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
        
        if self.mode in [Mode.TRAIN, Mode.TEST]:
            coco = self.dataset.coco

        if self.mode in [Mode.PREDICT, Mode.TEST]:
            with open(self.attentions, 'rb') as f:
                attentions = pickle.load(f)

            attentions_dataset = tf.data.Dataset\
                .from_tensors(attentions)\
                .repeat()

        if self.mode == Mode.TRAIN:
            if self.active_class_ids is None:
                self.active_class_ids = [cat['id'] for cat in self.dataset.coco.dataset['categories']]

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

            active_dataloader_class_ids = [0]+[self.dataset.get_dataloader_class_id(id) for id in self.active_class_ids]
            num_classes = len(coco.dataset['categories'])+1
            active_classes = [1 if dataloader_class_id in active_dataloader_class_ids else 0 for dataloader_class_id in range(num_classes)]

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
                        .filter(lambda prn_img: not tf.reduce_all(tf.math.is_nan(prn_img)))\
                        .repeat()
                    prn_datasets.append(dataset)
            
            prn_dataset = tf.data.Dataset\
                .zip(tuple(prn_datasets))\
                .map(lambda *prn_images: tf.stack(prn_images, 0), num_parallel_calls=tf.data.AUTOTUNE)\
                .batch(self.config.PRN_BATCH_PER_GPU*self.config.GPU_COUNT)
            

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
                .map(lambda *datas: [datas],
                     num_parallel_calls=tf.data.AUTOTUNE)\
                .prefetch(tf.data.AUTOTUNE)


        elif self.mode == Mode.PREDICT:

            if isinstance(self.image_pathes, str):
                self.image_pathes = [self.image_pathes]
            
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
                .map(lambda *datas: [datas],
                     num_parallel_calls=tf.data.AUTOTUNE)\
                .prefetch(tf.data.AUTOTUNE)
        
        else:
            pathes = tf.data.Dataset\
                .from_tensor_slices([img['path'] for img in coco.dataset['images']])
            img_ids = tf.data.Dataset\
                .from_tensor_slices([img['id'] for img in coco.dataset['images']])


            self.data_loader = tf.data.Dataset\
                .zip((pathes,img_ids))\
                .shuffle(len(self.dataset))\
                .map(self.preproccessing_test, num_parallel_calls=tf.data.AUTOTUNE)\
                .batch(self.batch_size)\
            
            self.data_loader = tf.data.Dataset\
                .zip((self.data_loader, attentions_dataset))\
                .map(lambda datas, attentions: [*datas, attentions],
                     num_parallel_calls=tf.data.AUTOTUNE)\
                .map(lambda *datas: [datas],
                     num_parallel_calls=tf.data.AUTOTUNE)\
                .repeat()\
                .prefetch(tf.data.AUTOTUNE)


    @tf.function
    def preprocessing_prn(self, ann_id):
        image_id = tf.py_function(lambda id: self.dataset.coco.loadAnns(int(id))[0]['image_id'], (ann_id,),tf.int32)
        path = tf.py_function(lambda id: self.dataset.coco.loadImgs(int(id))[0]['path'], (image_id,),tf.string)
        bbox = tf.py_function(lambda id: self.dataset.coco.loadAnns(int(id))[0]['bbox'], (ann_id,),(tf.float32,tf.float32,tf.float32,tf.float32))
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