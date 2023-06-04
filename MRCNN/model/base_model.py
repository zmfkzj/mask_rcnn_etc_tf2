from multiprocessing import Manager, Process
import re
from copy import copy, deepcopy
from typing import Optional

import keras.api._v2.keras as keras
import keras.api._v2.keras.layers as KL
import keras.api._v2.keras.models as KM
import numpy as np
import pycocotools.mask as maskUtils
import tensorflow as tf
from pycocotools.cocoeval import COCOeval
from tensorflow.python.keras.saving.saved_model.utils import \
    no_automatic_dependency_tracking_scope

from MRCNN import utils
from MRCNN.config import Config
from MRCNN.data.dataset_old import Dataset
from MRCNN.data.input_datas import InputDatas
from MRCNN.enums import EvalType, Mode, TrainLayers
from MRCNN.metric import F1Score
from MRCNN.model.outputs import Outputs

from ..utils import LossWeight, compute_backbone_shapes, unmold_detections


class BaseModel(KM.Model):
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """
    def __init__(self, config:Config, name):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        super().__init__(name=name)
        self.config = config

        self.anchors = self.get_anchors(config.IMAGE_SHAPE)

        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")
        # for evaluation
    
        self.backbone = self.make_backbone_model(config)
    
    
    def make_backbone_model(self, config: Config):
        backbone:KM.Model = config.BACKBONE(input_tensor=KL.Input(shape=list(config.IMAGE_SHAPE), dtype=tf.float32),
                                                 include_top=False,
                                                 weights='imagenet')

        output_channels = (2048,1024,512,256)
        self.backbone_output_shapes = compute_backbone_shapes(config)[:-1][::-1]
        self.backbone_output_shapes = list(zip(*zip(*self.backbone_output_shapes), output_channels))
        idx = 0
        output_ids = []
        for i, layer in enumerate(backbone.layers[::-1]):
            if tuple(layer.output_shape[1:]) == tuple(self.backbone_output_shapes[idx]):
                output_ids.append(i)
                idx+=1
            if idx==len(self.backbone_output_shapes):
                break
        output_ids =output_ids[:4]

        backbone:KM.Model = config.BACKBONE(input_tensor=KL.Input(shape=(None,None,3), dtype=tf.float32),
                                                 include_top=False,
                                                 weights='imagenet')
        outputs = [layer.output for i, layer in enumerate(backbone.layers[::-1]) if i in output_ids][::-1]

        model = keras.Model(inputs=backbone.inputs, outputs=outputs)
        return model
        

    def set_trainable(self, train_layers:TrainLayers, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Pre-defined layer regular expressions
        layer_regex = train_layers.value

        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            print("Selecting layers to train")

        keras_model:KM.Model = keras_model or self

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if isinstance(layer, KM.Model):
                print("In model: ", layer.name)
                self.set_trainable(train_layers, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainable layer names
            if trainable and verbose > 0:
                print("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))


    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = compute_backbone_shapes(self.config)
        # Generate Anchors
        anchors = utils.generate_pyramid_anchors(
            self.config.RPN_ANCHOR_SCALES,
            self.config.RPN_ANCHOR_RATIOS,
            backbone_shapes,
            self.config.BACKBONE_STRIDES,
            self.config.RPN_ANCHOR_STRIDE)
        norm_anchor = utils.norm_boxes(anchors, image_shape[:2])
        return norm_anchor