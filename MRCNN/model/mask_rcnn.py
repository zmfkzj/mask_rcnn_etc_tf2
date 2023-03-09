import keras.api._v2.keras as keras
import keras.api._v2.keras.layers as KL
import tensorflow as tf
import tensorflow_models as tfm

from MRCNN.config import Config
from MRCNN.enums import EvalType
from MRCNN.loss import (MrcnnBboxLossGraph, MrcnnClassLossGraph,
                        MrcnnMaskLossGraph, RpnBboxLossGraph,
                        RpnClassLossGraph)
from MRCNN.model.base_model import BaseModel, Model
from MRCNN.model.faster_rcnn import FasterRcnn

from ..layer import DetectionLayer, MrcnnTarget, ProposalLayer
from ..model_utils.miscellenous_graph import DenormBoxesGraph, NormBoxesGraph
from . import FPN_mask


class MaskRcnn(FasterRcnn):
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """
    def __init__(self, config:Config):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        super(BaseModel, self).__init__(name='mask_rcnn')
        self.build_parts(config)
        super(FasterRcnn, self).__init__(config,EvalType.SEGM)
    

    def build_parts(self, config: Config):
        super().build_parts(config)
        #additional parts
        self.ROIAlign_mask = tfm.vision.layers.MultilevelROIAligner(config.MASK_POOL_SIZE, name="roi_align_mask")
        self.fpn_mask = FPN_mask(config.NUM_CLASSES)

    
    def predict_step(self, data):
        input_images, input_window, origin_image_shapes, pathes = data[0]
        detections, mrcnn_mask = self.predict_model([input_images, input_window])
        return detections,mrcnn_mask,origin_image_shapes, input_window, pathes


    def test_step(self, data):
        input_images, input_window, origin_image_shapes, image_ids = data[0]
        detections, mrcnn_mask = self.test_model([input_images, input_window])
        tf.py_function(self.build_coco_results, (image_ids, detections, origin_image_shapes, input_window, mrcnn_mask), ())
        return {}


    def train_step(self, data):
        resized_image, resized_boxes, minimize_masks, dataloader_class_ids,rpn_match, rpn_bbox, active_class_ids = data[0]
        with tf.GradientTape() as tape:
            rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss = \
                self.train_model([resized_image, resized_boxes, minimize_masks, dataloader_class_ids,rpn_match, rpn_bbox, active_class_ids], training=True)

            reg_losses = tf.add_n([keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
                            for w in self.train_model.trainable_weights if 'gamma' not in w.name and 'beta' not in w.name])
            
            losses = [reg_losses, 
                    rpn_class_loss    * self.loss_weights.rpn_class_loss, 
                    rpn_bbox_loss     * self.loss_weights.rpn_bbox_loss, 
                    class_loss        * self.loss_weights.mrcnn_class_loss, 
                    bbox_loss         * self.loss_weights.mrcnn_bbox_loss, 
                    mask_loss         * self.loss_weights.mrcnn_mask_loss]

        gradients = tape.gradient(losses, self.train_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.train_model.trainable_variables))

        losses = [loss/self.config.GPU_COUNT for loss in losses]
        mean_loss = tf.reduce_mean(losses)
        reg_losses, rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss = losses
        return {'mean_loss':mean_loss,'rpn_class_loss':rpn_class_loss, 'rpn_bbox_loss':rpn_bbox_loss, 'class_loss':class_loss, 'bbox_loss':bbox_loss, 'mask_loss':mask_loss, 'reg_losses':reg_losses, 'lr':self.optimizer.learning_rate}


    def make_predict_model(self):
        input_image = KL.Input(self.config.IMAGE_SHAPE, dtype=tf.uint8, name='predict_input_image')
        input_window = KL.Input(shape=(4,), name="predict_input_window")

        backbone_output = self.backbone(input_image)
        P2,P3,P4,P5,P6 = self.neck(*backbone_output)
        
        P2 = tf.ensure_shape(P2, (None,)+self.backbone_output_shapes[-1][:2]+(256,))
        P3 = tf.ensure_shape(P3, (None,)+self.backbone_output_shapes[-2][:2]+(256,))
        P4 = tf.ensure_shape(P4, (None,)+self.backbone_output_shapes[-3][:2]+(256,))
        P5 = tf.ensure_shape(P5, (None,)+self.backbone_output_shapes[-4][:2]+(256,))

        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = {'2':P2, '3':P3, '4':P4, '5':P5}

        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(self.rpn(p))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        outputs = list(zip(*layer_outputs))
        outputs = [c(list(o)) for o, c in zip(outputs, [KL.Concatenate(axis=1, name=n) for n in ["rpn_class_logits", "rpn_class", "rpn_bbox"]])]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        batch_size = tf.shape(input_image)[0]
        anchors = tf.broadcast_to(self.anchors, tf.concat([(batch_size,),tf.shape(self.anchors)],-1))
        proposal_count = self.config.POST_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(nms_threshold=self.config.RPN_NMS_THRESHOLD, name="predict_ROI", config=self.config)([rpn_class, rpn_bbox, anchors, proposal_count])
        
        _rpn_rois = KL.Lambda(lambda r: 
                          tf.cast(tf.vectorized_map(lambda x: 
                                                    DenormBoxesGraph()(x,list(self.config.IMAGE_SHAPE)[:2]),r), tf.float32))(rpn_rois)
        roi_cls_feature = self.ROIAlign_classifier(mrcnn_feature_maps, _rpn_rois)

        # Network Heads
        # Proposal classifier and BBox regressor heads
        mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.fpn_classifier(roi_cls_feature, training=False)
        detections = DetectionLayer(self.config, name="predict_mrcnn_detection")(rpn_rois, mrcnn_class, mrcnn_bbox, self.config.IMAGE_SHAPE, input_window)

        # Create masks for detections
        detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
        # roi_seg_feature = self.ROIAlign_mask(detection_boxes, self.config.IMAGE_SHAPE, mrcnn_feature_maps)
        _detection_boxes = KL.Lambda(lambda r: 
                          tf.cast(tf.vectorized_map(lambda x: 
                                                    DenormBoxesGraph()(x,list(self.config.IMAGE_SHAPE)[:2]),r), tf.float32))(detection_boxes)
        roi_seg_feature = self.ROIAlign_mask(mrcnn_feature_maps, _detection_boxes)
        mrcnn_mask = self.fpn_mask(roi_seg_feature, training=False)

        model = keras.Model([input_image, input_window],
                            [detections, mrcnn_mask],
                            name='predict_mask_rcnn')
        return model

    
    def make_test_model(self):
        return self.make_predict_model()
    

    def make_train_model(self):
        input_image = KL.Input( shape=[None, None, self.config.IMAGE_SHAPE[2]], name="train_input_image")
        active_class_ids = KL.Input(shape=[self.config.NUM_CLASSES], name="train_input_class_ids")
        # RPN GT
        input_rpn_match = KL.Input( shape=[None, 1], name="train_input_rpn_match", dtype=tf.int64)
        input_rpn_bbox = KL.Input( shape=[None, 4], name="train_input_rpn_bbox", dtype=tf.float32)

        # Detection GT (class IDs, bounding boxes, and masks)
        # 1. GT Class IDs (zero padded)
        input_gt_class_ids = KL.Input( shape=[None], name="train_input_gt_class_ids", dtype=tf.int64)
        # 2. GT Boxes in pixels (zero padded)
        # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
        input_gt_boxes = KL.Input( shape=[None, 4], name="train_input_gt_boxes", dtype=tf.float32)
        # Normalize coordinates
        gt_boxes = NormBoxesGraph()(input_gt_boxes, tf.shape(input_image)[1:3])
        # 3. GT Masks (zero padded)
        # [batch, height, width, MAX_GT_INSTANCES]
        input_gt_masks = KL.Input( shape=[self.config.MINI_MASK_SHAPE[0], self.config.MINI_MASK_SHAPE[1], None], name="train_input_gt_masks", dtype=bool)

        backbone_output = self.backbone(input_image)
        P2,P3,P4,P5,P6 = self.neck(*backbone_output)
        
        P2 = tf.ensure_shape(P2, (None,)+self.backbone_output_shapes[-1][:2]+(256,))
        P3 = tf.ensure_shape(P3, (None,)+self.backbone_output_shapes[-2][:2]+(256,))
        P4 = tf.ensure_shape(P4, (None,)+self.backbone_output_shapes[-3][:2]+(256,))
        P5 = tf.ensure_shape(P5, (None,)+self.backbone_output_shapes[-4][:2]+(256,))

        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = {'2':P2, '3':P3, '4':P4, '5':P5}

        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(self.rpn(p))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        outputs = list(zip(*layer_outputs))
        outputs = [c(list(o)) for o, c in zip(outputs, [KL.Concatenate(axis=1, name=n) for n in ["rpn_class_logits", "rpn_class", "rpn_bbox"]])]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        batch_size = tf.shape(input_image)[0]
        anchors = tf.broadcast_to(self.anchors, tf.concat([(batch_size,),tf.shape(self.anchors)],-1))
        proposal_count = self.config.POST_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(nms_threshold=self.config.RPN_NMS_THRESHOLD, name="train_ROI", config=self.config)([rpn_class, rpn_bbox, anchors, proposal_count])


        # Generate detection targets
        # Subsamples proposals and generates target outputs for training
        # Note that proposal class IDs, gt_boxes, and gt_masks are zero
        # padded. Equally, returned rois and targets are zero padded.
        rois, target_class_ids, target_bbox, target_mask =\
            MrcnnTarget(self.config, name="train_proposal_targets")([rpn_rois, input_gt_class_ids, gt_boxes, input_gt_masks])
        

        _rois = KL.Lambda(lambda r: 
                          tf.cast(tf.vectorized_map(lambda x: 
                                                    DenormBoxesGraph()(x,list(self.config.IMAGE_SHAPE)[:2]),r), tf.float32))(rois)
        roi_cls_features = self.ROIAlign_classifier(mrcnn_feature_maps, _rois)
        roi_mask_features = self.ROIAlign_mask(mrcnn_feature_maps, _rois)

        # Network Heads
        # TODO: verify that this handles zero padded ROIs
        mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.fpn_classifier(roi_cls_features)
        mrcnn_mask = self.fpn_mask(roi_mask_features)


        # Losses
        rpn_class_loss = RpnClassLossGraph(name="rpn_class_loss")( input_rpn_match, rpn_class_logits)
        rpn_bbox_loss = RpnBboxLossGraph(name="rpn_bbox_loss")(input_rpn_bbox, input_rpn_match, rpn_bbox, batch_size)
        class_loss = MrcnnClassLossGraph(name="mrcnn_class_loss")(target_class_ids, mrcnn_class_logits, active_class_ids)
        bbox_loss = MrcnnBboxLossGraph(name="mrcnn_bbox_loss")(target_bbox, target_class_ids, mrcnn_bbox)
        mask_loss = MrcnnMaskLossGraph(name="mrcnn_mask_loss")(target_mask, target_class_ids, mrcnn_mask)

        # Model
        inputs = [input_image, input_gt_boxes, input_gt_masks, input_gt_class_ids, input_rpn_match, input_rpn_bbox, active_class_ids]
        outputs = [rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]

        model = keras.Model(inputs, outputs, name='train_mask_rcnn')
        return model