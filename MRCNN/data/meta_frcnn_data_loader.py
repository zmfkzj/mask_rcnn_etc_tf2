import os
from pydantic.dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pickle
import tensorflow as tf

from MRCNN.data import frcnn_data_loader
from MRCNN import PydanticConfig
from MRCNN.data.input_datas import InputDatas
from MRCNN.enums import Mode
from MRCNN.utils import (compute_backbone_shapes,
                         generate_pyramid_anchors)
from pycocotools import mask as maskUtils


@dataclass(config=PydanticConfig)
class DataLoader(frcnn_data_loader.DataLoader):

    phase:int=2
    attentions:Union[str,np.ndarray,tf.Tensor,None] = None
    prn_batch_size:Optional[int] = None

    def __hash__(self) -> int:
        return hash((tuple(self.config.ACTIVE_CLASS_IDS) if self.config.ACTIVE_CLASS_IDS is not None else None, 
                     self.mode, 
                     tuple(self.image_pathes) if self.image_pathes is not None else None, 
                     self.dataset,
                     tuple(self.novel_classes),
                     self.phase))
    

    def make_dataloader(self):
        if self.phase==1:
            self.dataset.set_dataloader_class_list(self.config.NOVEL_CLASSES)
            self.config.set_phase(1)
        
        self.novel_classes = tuple(self.config.NOVEL_CLASSES)
        if self.mode!=Mode.PREDICT:
            if self.config.SHOTS > self.dataset.min_class_count:
                ValueError('SHOTS must be less than min_class_count')
        

        if self.mode==Mode.PRN:
            self.data_loader = self.make_train_dataloader()
        else: 
            super().make_dataloader()
        
        if self.mode in [Mode.PRN, Mode.TRAIN]:
            with self.config.STRATEGY.scope():
                self.data_loader = self.config.STRATEGY.experimental_distribute_dataset(self.data_loader)


            

    def make_predict_dataloader(self):
        data_loader = super().make_predict_dataloader()

        if isinstance(self.attentions,str):
            with open(self.attentions, 'rb') as f:
                attentions = pickle.load(f)
        elif self.attentions is None:
            ValueError('argument attentions is necessary.')

        attentions_dataset = tf.data.Dataset\
            .from_tensors(attentions)\
            .repeat()\
        
        data_loader = tf.data.Dataset\
            .zip((data_loader, attentions_dataset))\
            .map(lambda datas, attentions: [InputDatas(self.config,**datas[0]).update_attentions(attentions).to_dict()],
                num_parallel_calls=tf.data.AUTOTUNE)\
            .prefetch(tf.data.AUTOTUNE)
        return data_loader


    def make_test_dataloader(self):
        data_loader = super().make_test_dataloader()

        if self.attentions is not None:
            if isinstance(self.attentions,str):
                with open(self.attentions, 'rb') as f:
                    attentions = pickle.load(f)
            else:
                attentions = self.attentions

            attentions_dataset = tf.data.Dataset\
                .from_tensors(attentions)\
                .repeat()\

            data_loader = tf.data.Dataset\
                .zip((data_loader, attentions_dataset))\
                .map(lambda datas, attentions: [InputDatas(self.config,**datas[0]).update_attentions(attentions).to_dict()],
                    num_parallel_calls=tf.data.AUTOTUNE)\
                .prefetch(tf.data.AUTOTUNE)
        
        return data_loader
    

    def make_train_dataloader(self):
        data_loader = super().make_train_dataloader()

        self.prn_batch_size = self.config.PRN_BATCH_SIZE if self.prn_batch_size is None else self.prn_batch_size
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
        
        data_loader = tf.data.Dataset\
            .zip((data_loader, prn_dataset))\
            .map(lambda datas, prn_images: [InputDatas(self.config, **datas[0]).update_prn_images(prn_images).to_dict()],
                    num_parallel_calls=tf.data.AUTOTUNE)\
            .prefetch(tf.data.AUTOTUNE)
        
        return data_loader


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

        mask = tf.py_function(lambda bbox, h, w:self.bbox_to_mask(bbox, h, w),(resized_bbox,*self.config.PRN_IMAGE_SIZE),tf.bool)
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
         

    def bbox_to_rle(self, bbox, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        y1,x1,y2,x2 = bbox
        segm = [[x1,y1,x2,y1,x2,y2,x1,y2]]
        rles = maskUtils.frPyObjects(segm, height, width)
        rle = maskUtils.merge(rles)
        return rle

    def bbox_to_mask(self, bbox, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.bbox_to_rle(bbox, height, width)
        m = maskUtils.decode(rle)
        return m


    def set_active_class_ids(self):
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
        return self.active_class_ids

