from typing import Callable
import keras.api._v2.keras as keras
import keras.api._v2.keras.layers as KL
import tensorflow as tf
import tensorflow_models as tfm

from MRCNN.config import Config
from MRCNN.data.dataset import Dataset
from MRCNN.enums import EvalType, Mode
from MRCNN.loss import (MrcnnBboxLossGraph, MrcnnClassLossGraph,
                        RpnBboxLossGraph, RpnClassLossGraph)
from MRCNN.model.base_model import BaseModel

from ..layer import DetectionLayer, FrcnnTarget
from ..model_utils.miscellenous_graph import DenormBoxesGraph, NormBoxesGraph
from . import RPN, FPN_classifier, Neck


class FasterRcnn(BaseModel):
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """
    def __init__(self, config:Config, dataset:Dataset, name='faster_rcnn'):
        """
        config: A Sub-class of the Config class
        """
        super().__init__(config, dataset, name=name)

        self.ROIAlign_classifier = tfm.vision.layers.MultilevelROIAligner(config.POOL_SIZE, name="roi_align_classifier")
        self.fpn_classifier = FPN_classifier(config.POOL_SIZE, self.num_classes, fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

    
    def train_step(self, data):
        input_images, input_gt_boxes, input_gt_class_ids, input_rpn_match, input_rpn_bbox, input_gt_masks= data
        batch_size = tf.shape(input_images)[0]

        with tf.GradientTape() as tape:

            rpn_rois, mrcnn_feature_maps, rpn_class_logits, rpn_bbox = self(input_images)
            target_class_ids, mrcnn_class_logits, target_bbox, mrcnn_bbox = \
                self.forward_train(rpn_rois, mrcnn_feature_maps, input_gt_class_ids, input_gt_boxes)


            # Losses
            rpn_class_loss = RpnClassLossGraph(name="rpn_class_loss")( input_rpn_match, rpn_class_logits)
            rpn_bbox_loss = RpnBboxLossGraph(name="rpn_bbox_loss")(input_rpn_bbox, input_rpn_match, rpn_bbox, batch_size)
            class_loss = MrcnnClassLossGraph(name="mrcnn_class_loss")(target_class_ids, mrcnn_class_logits)
            bbox_loss = MrcnnBboxLossGraph(name="mrcnn_bbox_loss")(target_bbox, target_class_ids, mrcnn_bbox)

            reg_losses = tf.add_n([tf.cast(keras.regularizers.l2(self.config.WEIGHT_DECAY)(w), tf.float16) / tf.cast(tf.size(w), tf.float16)
                            for w in self.trainable_weights if 'gamma' not in w.name and 'beta' not in w.name])
            
            losses = [reg_losses, 
                    rpn_class_loss    * self.config.LOSS_WEIGHTS['rpn_class_loss'], 
                    rpn_bbox_loss     * self.config.LOSS_WEIGHTS['rpn_bbox_loss'], 
                    class_loss        * self.config.LOSS_WEIGHTS['mrcnn_class_loss'], 
                    bbox_loss         * self.config.LOSS_WEIGHTS['mrcnn_bbox_loss']]
            losses = [loss / self.config.GPU_COUNT for loss in losses]
            scaled_losses = [self.optimizer.get_scaled_loss(loss) for loss in losses]

        scaled_grad = tape.gradient(scaled_losses, self.trainable_variables)
        grad = self.optimizer.get_unscaled_gradients(scaled_grad)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

        mean_loss = tf.reduce_mean(losses)
        reg_losses, rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss = losses
        return {'mean_loss':mean_loss,'rpn_class_loss':rpn_class_loss, 'rpn_bbox_loss':rpn_bbox_loss, 'class_loss':class_loss, 'bbox_loss':bbox_loss, 'reg_losses':reg_losses, 'lr':self.optimizer.lr}

    
    def test_step(self, data):
        input_images, input_window, origin_image_shapes, image_ids = data

        rpn_rois, mrcnn_feature_maps, rpn_class_logits, rpn_bbox = self(input_images)
        detections = self.forward_predict_test(rpn_rois, mrcnn_feature_maps, input_window)
        tf.py_function(self.build_coco_results, (image_ids, 
                                                 detections, 
                                                 origin_image_shapes, 
                                                 input_window), ())
        return {}


    def predict_step(self, data):
        input_images, input_window, origin_image_shapes, pathes = data
        rpn_rois, mrcnn_feature_maps, rpn_class_logits, rpn_bbox = self(input_images)
        detections = self.forward_predict_test(rpn_rois, mrcnn_feature_maps, input_window)
        tf.py_function(self.build_detection_results, (pathes, 
                                                      detections, 
                                                      origin_image_shapes, 
                                                      input_window), ())

    
    @tf.function
    def forward_predict_test(self, rpn_rois, mrcnn_feature_maps, input_window):
        _rpn_rois = KL.Lambda(lambda r: 
                          tf.cast(tf.vectorized_map(lambda x: 
                                                    DenormBoxesGraph()(x,list(self.config.IMAGE_SHAPE)[:2]),r), tf.float16))(rpn_rois)
        roi_cls_feature = self.ROIAlign_classifier(mrcnn_feature_maps, _rpn_rois)

        # Network Heads
        # Proposal classifier and BBox regressor heads
        mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.fpn_classifier(roi_cls_feature, training=False)
        detections = DetectionLayer(self.config, name="detection")(rpn_rois, mrcnn_class, mrcnn_bbox, self.config.IMAGE_SHAPE, input_window)
        return detections



    @tf.function
    def forward_train(self, rpn_rois, mrcnn_feature_maps, input_gt_class_ids, input_gt_boxes):

        # Normalize coordinates
        gt_boxes = NormBoxesGraph()(input_gt_boxes, self.config.IMAGE_SHAPE[:2])

        # Generate detection targets
        # Subsamples proposals and generates target outputs for training
        # Note that proposal class IDs, gt_boxes, and gt_masks are zero
        # padded. Equally, returned rois and targets are zero padded.
        rois, target_class_ids, target_bbox =\
            FrcnnTarget(self.config, name="proposal_targets")([rpn_rois, input_gt_class_ids, gt_boxes])
        

        _rois = KL.Lambda(lambda r: 
                          tf.cast(tf.vectorized_map(lambda x: 
                                                    DenormBoxesGraph()(x,list(self.config.IMAGE_SHAPE)[:2]),r), tf.float16))(rois)
        roi_cls_features = self.ROIAlign_classifier(mrcnn_feature_maps, _rois)

        # Network Heads
        # TODO: verify that this handles zero padded ROIs
        mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.fpn_classifier(roi_cls_features)
        return target_class_ids, mrcnn_class_logits, target_bbox, mrcnn_bbox