import os
import re
import datetime
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
import numpy as np

from MRCNN.model.mask_rcnn import MaskRCNN
from MRCNN.config import Config
from MRCNN.loss import mrcnn_bbox_loss_graph, mrcnn_class_loss_graph, mrcnn_mask_loss_graph, rpn_bbox_loss_graph, rpn_class_loss_graph
from MRCNN.data_generator import data_generator, compose_image_meta, mold_image
from MRCNN.layer.roialign import parse_image_meta_graph


class Trainer:
    def __init__(self, model:MaskRCNN, config, train_dataset, layers,
                        val_dataset = None, test_dataset = None,
                        optimizer = keras.optimizers.SGD(),
                        logs_dir='logs/'):
        train_dataset = 
        

        mirrored_strategy = tf.distribute.MirroredStrategy(divices=[f'/gpu:{gpu_id}' for gpu_id in config.CPUS])
        with mirrored_strategy.scope():
            self.model = model(config, logs_dir)
            self.train_dataset = mirrored_strategy.experimental_distribute_dataset(
                                    tf.data.Dataset.from_generator(self.load_dataset(train_dataset)).prefetch())
            if val_dataset is not None:
                self.val_dataset = mirrored_strategy.experimental_distribute_dataset(
                                    tf.data.Dataset.from_generator(self.load_dataset(val_dataset)).prefetch())
            if test_dataset is not None:
                self.test_dataset = mirrored_strategy.experimental_distribute_dataset(
                                    tf.data.Dataset.from_generator(self.load_dataset(test_dataset)).prefetch())
            
            self.optimizer = optimizer
        
        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        self.model.set_trainable(layers)

    @tf.function
    def train_step(self, dist_images, dist_labels):
        def step_fn(images, labels):
            with tf.GradientTape() as tape:
                output = self.model(images, input_image_meta, 
                                    input_anchors=None, 
                                    input_gt_class_ids=None, 
                                    input_gt_boxes=None, 
                                    input_gt_masks=None, 
                                    input_rois = None,
                                    training=True)
                active_class_ids = KL.Lambda( lambda x: parse_image_meta_graph(x)["active_class_ids"])(input_image_meta)
                loss = self.cal_loss(active_class_ids, *output)
    
    @tf.function
    def cal_loss(self, active_class_ids, rpn_class_logits, rpn_bbox, target_class_ids, mrcnn_class_logits, target_bbox, mrcnn_bbox, target_mask, mrcnn_mask):
        input_rpn_match = 
        input_rpn_bbox = 

        rpn_class_loss = tf.reduce_mean(self.config.LOSS_WEIGHTS.get('rpn_class_loss', 1.)
                                        * rpn_class_loss_graph(*[input_rpn_match, rpn_class_logits]), keepdims=True)
        rpn_bbox_loss = tf.reduce_mean(self.config.LOSS_WEIGHTS.get('rpn_bbox_loss', 1.) 
                                        * rpn_bbox_loss_graph(config,*[input_rpn_bbox, input_rpn_match, rpn_bbox]), keepdims=True)
        class_loss = tf.reduce_mean(self.config.LOSS_WEIGHTS.get('mrcnn_class_loss', 1.) 
                                    * mrcnn_class_loss_graph(*[target_class_ids, mrcnn_class_logits, active_class_ids]), keepdims=True)
        bbox_loss = tf.reduce_mean(self.config.LOSS_WEIGHTS.get('mrcnn_bbox_loss', 1.) 
                                    * mrcnn_bbox_loss_graph(*[target_bbox, target_class_ids, mrcnn_bbox]), keepdims=True)
        mask_loss = tf.reduce_mean(self.config.LOSS_WEIGHTS.get('mrcnn_mask_loss', 1.) 
                                    * mrcnn_mask_loss_graph(*[target_mask, target_class_ids, mrcnn_mask]), keepdims=True)
        
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
                                    for w in self.model.trainable_weights
                                    if 'gamma' not in w.name and 'beta' not in w.name]


        return tf.reduce_sum([rpn_bbox_loss, rpn_class_loss, class_loss, bbox_loss, mask_loss, tf.add_n(reg_losses)])
    
    @staticmethod
    def load_dataset(dataset, config, augmentation=None, no_augmentation_sources=None, training=False):
        dataset.prepare()
        if training:
            return data_generator(dataset, config, shuffle=True, augmentation=augmentation, batch_size=config.BATCH_SIZE, no_augmentation_sources=no_augmentation_sources)
        else:
            return data_generator(dataset, config, shuffle=True, batch_size=config.BATCH_SIZE)
