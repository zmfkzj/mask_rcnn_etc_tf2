from dataclasses import InitVar
from typing import Optional
import tensorflow as tf
from pydantic.dataclasses import dataclass
from pydantic import BaseModel
from MRCNN import PydanticConfig

from MRCNN.config import Config


@dataclass(config=PydanticConfig)
class InputDatas:
    config:InitVar[Config]
    anchor_size:InitVar[Optional[int]] = None

    input_gt_boxes: Optional[tf.Tensor] = None
    dataloader_class_ids: Optional[tf.Tensor] = None
    rpn_match: Optional[tf.Tensor] = None
    rpn_bbox: Optional[tf.Tensor] = None
    active_class_ids: Optional[tf.Tensor] = None
    prn_images: Optional[tf.Tensor] = None
    pathes: Optional[tf.Tensor] = None
    input_images: Optional[tf.Tensor] = None
    input_window: Optional[tf.Tensor] = None
    origin_image_shapes: Optional[tf.Tensor] = None
    image_ids: Optional[tf.Tensor] = None
    attentions: Optional[tf.Tensor] = None
    input_gt_masks: Optional[tf.Tensor] = None

    def __post_init__(self, config:Config, anchor_size:Optional[int]=None):
        anchor_size = anchor_size or 0

        self.input_gt_boxes = self.input_gt_boxes \
            if self.input_gt_boxes is not None \
            else tf.zeros([config.TRAIN_BATCH_SIZE,config.MAX_GT_INSTANCES,4],dtype=tf.float32)
        self.dataloader_class_ids = self.dataloader_class_ids \
            if self.dataloader_class_ids is not None\
            else tf.zeros([config.TRAIN_BATCH_SIZE,config.MAX_GT_INSTANCES],dtype=tf.int64)
        self.rpn_match = self.rpn_match if self.rpn_match is not None\
            else tf.zeros([config.TRAIN_BATCH_SIZE,anchor_size], dtype=tf.int64)
        self.rpn_bbox = self.rpn_bbox if self.rpn_bbox is not None\
            else tf.zeros([config.TRAIN_BATCH_SIZE,config.RPN_TRAIN_ANCHORS_PER_IMAGE,4], dtype=tf.float32)
        self.active_class_ids = self.active_class_ids if self.active_class_ids is not None\
            else tf.zeros([config.TRAIN_BATCH_SIZE,config.NUM_CLASSES], dtype=tf.int32)
        self.prn_images = self.prn_images if self.prn_images is not None\
            else tf.zeros([config.PRN_BATCH_SIZE,config.NUM_CLASSES-1,*config.PRN_IMAGE_SIZE, 4], dtype=tf.float32)
        self.pathes = self.pathes if self.pathes is not None\
            else tf.zeros([config.TRAIN_BATCH_SIZE],dtype=tf.string)
        self.input_images = self.input_images if self.input_images is not None\
            else tf.zeros([config.TRAIN_BATCH_SIZE, *config.IMAGE_SHAPE], dtype=tf.float32)
        self.input_window = self.input_window if self.input_window is not None\
            else tf.zeros([config.TRAIN_BATCH_SIZE,4], dtype=tf.float32)
        self.origin_image_shapes = self.origin_image_shapes if self.origin_image_shapes is not None\
            else tf.zeros([config.TRAIN_BATCH_SIZE,2], dtype=tf.int32)
        self.image_ids = self.image_ids if self.image_ids is not None\
            else tf.zeros([config.TRAIN_BATCH_SIZE], dtype=tf.int32)
        self.attentions = self.attentions if self.attentions is not None\
            else tf.zeros([config.NUM_CLASSES,config.FPN_CLASSIF_FC_LAYERS_SIZE], dtype=tf.float32)
        self.input_gt_masks = self.input_gt_masks if self.input_gt_masks is not None\
            else tf.zeros([config.TRAIN_BATCH_SIZE,config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1],config.MAX_GT_INSTANCES], dtype=tf.bool)


    def to_dict(self):
        return {
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
            'attentions': self.attentions,
            'input_gt_masks': self.input_gt_masks
        }