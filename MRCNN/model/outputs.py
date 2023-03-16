from dataclasses import InitVar
from typing import Optional
import tensorflow as tf
from pydantic.dataclasses import dataclass

from MRCNN import PydanticConfig
from MRCNN.config import Config


@dataclass(config=PydanticConfig)
class OutputsArgs:
    config:InitVar[Optional[Config]] = None

    detections: Optional[ tf.Tensor ] = None
    attentions: Optional[ tf.Tensor ] = None
    attentions_logits: Optional[ tf.Tensor ] = None
    rpn_class_loss: Optional[ tf.Tensor ] = None
    rpn_bbox_loss: Optional[ tf.Tensor ] = None
    class_loss: Optional[ tf.Tensor ] = None
    bbox_loss: Optional[ tf.Tensor ] = None
    meta_loss: Optional[ tf.Tensor ] = None
    masks: Optional[ tf.Tensor ] = None
    mask_loss: Optional[ tf.Tensor ] = None


    def __post_init__(self, config: Config):
        self.detections = self.detections\
            if self.detections is not None\
            else tf.zeros([config.TRAIN_IMAGES_PER_GPU,config.MAX_GT_INSTANCES,6], dtype=tf.float32)
        self.attentions = self.attentions\
            if self.attentions is not None\
            else tf.zeros([config.NUM_CLASSES, config.FPN_CLASSIF_FC_LAYERS_SIZE], tf.float32)
        self.attentions_logits = self.attentions_logits\
            if self.attentions_logits is not None\
            else tf.zeros([config.NUM_CLASSES, config.FPN_CLASSIF_FC_LAYERS_SIZE], tf.float32)
        self.rpn_class_loss = self.rpn_class_loss\
            if self.rpn_class_loss is not None\
            else tf.constant(0., tf.float32)
        self.rpn_bbox_loss = self.rpn_bbox_loss\
            if self.rpn_bbox_loss is not None\
            else tf.constant(0., tf.float32)
        self.class_loss = self.class_loss\
            if self.class_loss is not None\
            else tf.constant(0., tf.float32)
        self.bbox_loss = self.bbox_loss\
            if self.bbox_loss is not None\
            else tf.constant(0., tf.float32)
        self.meta_loss = self.meta_loss\
            if self.meta_loss is not None\
            else tf.constant(0., tf.float32)
        self.masks = self.masks\
            if self.masks is not None\
            else tf.zeros([config.TRAIN_IMAGES_PER_GPU,
                           config.MAX_GT_INSTANCES,
                           config.MASK_POOL_SIZE, 
                           config.MASK_POOL_SIZE, 
                           config.NUM_CLASSES], dtype=tf.float32)
        self.mask_loss = self.mask_loss\
            if self.mask_loss is not None\
            else tf.constant(0., tf.float32)

    def to_dict(self):
        return {
            'detections': self.detections,
            'attentions': self.attentions,
            'attentions_logits': self.attentions_logits,
            'rpn_class_loss': self.rpn_class_loss,
            'rpn_bbox_loss': self.rpn_bbox_loss,
            'class_loss': self.class_loss,
            'bbox_loss': self.bbox_loss,
            'meta_loss': self.meta_loss,
            'masks':self.masks,
            'mask_loss':self.mask_loss
        }



class Outputs(tf.experimental.ExtensionType):
    detections: tf.Tensor
    attentions: tf.Tensor
    attentions_logits: tf.Tensor
    rpn_class_loss: tf.Tensor
    rpn_bbox_loss: tf.Tensor
    class_loss: tf.Tensor
    bbox_loss: tf.Tensor
    meta_loss: tf.Tensor
    masks: tf.Tensor
    mask_loss: tf.Tensor

    @staticmethod
    def from_args(args:OutputsArgs):
        return Outputs(**args.to_dict())
    
    def update_detections(self, new):
        args = OutputsArgs(**self.__dict__)
        args.detections = new
        return Outputs.from_args(args)
    
    def update_attentions(self, new):
        args = OutputsArgs(**self.__dict__)
        args.attentions = new
        return Outputs.from_args(args)
    
    def update_attentions_logits(self, new):
        args = OutputsArgs(**self.__dict__)
        args.attentions_logits = new
        return Outputs.from_args(args)
    
    def update_rpn_class_loss(self, new):
        args = OutputsArgs(**self.__dict__)
        args.rpn_class_loss = new
        return Outputs.from_args(args)
    
    def update_rpn_bbox_loss(self, new):
        args = OutputsArgs(**self.__dict__)
        args.rpn_bbox_loss = new
        return Outputs.from_args(args)
    
    def update_class_loss(self, new):
        args = OutputsArgs(**self.__dict__)
        args.class_loss = new
        return Outputs.from_args(args)
    
    def update_bbox_loss(self, new):
        args = OutputsArgs(**self.__dict__)
        args.bbox_loss = new
        return Outputs.from_args(args)
    
    def update_meta_loss(self, new):
        args = OutputsArgs(**self.__dict__)
        args.meta_loss = new
        return Outputs.from_args(args)
    
    def update_masks(self, new):
        args = OutputsArgs(**self.__dict__)
        args.masks = new
        return Outputs.from_args(args)
    
    def update_mask_loss(self, new):
        args = OutputsArgs(**self.__dict__)
        args.mask_loss = new
        return Outputs.from_args(args)