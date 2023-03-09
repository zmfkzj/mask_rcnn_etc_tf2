import keras.api._v2.keras as keras
import keras.api._v2.keras.layers as KL
import tensorflow as tf
import tensorflow_models as tfm

from MRCNN.config import Config
from MRCNN.enums import EvalType
from MRCNN.loss import (MrcnnBboxLossGraph, MrcnnClassLossGraph,
                        RpnBboxLossGraph, RpnClassLossGraph)
from MRCNN.model.base_model import BaseModel

from ..layer import DetectionLayer, FrcnnTarget, ProposalLayer
from ..model_utils.miscellenous_graph import DenormBoxesGraph, NormBoxesGraph
from . import RPN, FPN_classifier, Neck


class FasterRcnn(BaseModel):
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """
    def __init__(self, config:Config):
        """
        config: A Sub-class of the Config class
        """
        super(BaseModel, self).__init__(name='faster_rcnn')
        self.build_parts(config)
        super().__init__(config, EvalType.BBOX)

    
    def build_parts(self, config: Config):
        super().build_parts(config)
        #additional parts
        self.neck = Neck(config)
        self.rpn = RPN(config.RPN_ANCHOR_STRIDE, len(config.RPN_ANCHOR_RATIOS), name='rpn_model')

        self.ROIAlign_classifier = tfm.vision.layers.MultilevelROIAligner(config.POOL_SIZE, name="roi_align_classifier")
        self.fpn_classifier = FPN_classifier(config.POOL_SIZE, config.NUM_CLASSES, fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

    
    def predict_step(self, data):
        input_images, input_window, origin_image_shapes, pathes = data[0]
        detections = self.predict_model([input_images, input_window])
        return detections,origin_image_shapes, input_window, pathes


    def test_step(self, data):
        input_images, input_window, origin_image_shapes, image_ids = data[0]
        detections = self.test_model([input_images, input_window])
        tf.py_function(self.build_coco_results, (image_ids, detections, origin_image_shapes, input_window), ())
        return {}


    def train_step(self, data):
        resized_image, resized_boxes, dataloader_class_ids,rpn_match, rpn_bbox, active_class_ids = data[0]
        with tf.GradientTape() as tape:
            rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss = \
                self.train_model([resized_image, resized_boxes, dataloader_class_ids,rpn_match, rpn_bbox, active_class_ids], training=True)

            reg_losses = tf.add_n([keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
                            for w in self.train_model.trainable_weights if 'gamma' not in w.name and 'beta' not in w.name])
            
            losses = [reg_losses, 
                    rpn_class_loss    * self.loss_weights.rpn_class_loss, 
                    rpn_bbox_loss     * self.loss_weights.rpn_bbox_loss, 
                    class_loss        * self.loss_weights.mrcnn_class_loss, 
                    bbox_loss         * self.loss_weights.mrcnn_bbox_loss]

        gradients = tape.gradient(losses, self.train_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.train_model.trainable_variables))

        losses = [loss/self.config.GPU_COUNT for loss in losses]
        mean_loss = tf.reduce_mean(losses)
        reg_losses, rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss = losses
        return {'mean_loss':mean_loss,'rpn_class_loss':rpn_class_loss, 'rpn_bbox_loss':rpn_bbox_loss, 'class_loss':class_loss, 'bbox_loss':bbox_loss, 'reg_losses':reg_losses, 'lr':self.optimizer.learning_rate}


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
        detections = DetectionLayer(self.config, name="detection")(rpn_rois, mrcnn_class, mrcnn_bbox, self.config.IMAGE_SHAPE, input_window)

        model = keras.Model([input_image, input_window],
                            detections,
                            name='predict_model')
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
        rois, target_class_ids, target_bbox =\
            FrcnnTarget(self.config, name="proposal_targets")([rpn_rois, input_gt_class_ids, gt_boxes])
        

        _rois = KL.Lambda(lambda r: 
                          tf.cast(tf.vectorized_map(lambda x: 
                                                    DenormBoxesGraph()(x,list(self.config.IMAGE_SHAPE)[:2]),r), tf.float32))(rois)
        roi_cls_features = self.ROIAlign_classifier(mrcnn_feature_maps, _rois)

        # Network Heads
        # TODO: verify that this handles zero padded ROIs
        mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.fpn_classifier(roi_cls_features)


        # Losses
        rpn_class_loss = RpnClassLossGraph(name="rpn_class_loss")( input_rpn_match, rpn_class_logits)
        rpn_bbox_loss = RpnBboxLossGraph(name="rpn_bbox_loss")(input_rpn_bbox, input_rpn_match, rpn_bbox, batch_size)
        class_loss = MrcnnClassLossGraph(name="mrcnn_class_loss")(target_class_ids, mrcnn_class_logits, active_class_ids)
        bbox_loss = MrcnnBboxLossGraph(name="mrcnn_bbox_loss")(target_bbox, target_class_ids, mrcnn_bbox)

        # Model
        inputs = [input_image, input_gt_boxes, input_gt_class_ids, input_rpn_match, input_rpn_bbox, active_class_ids]
        outputs = [rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss]

        model = keras.Model(inputs, outputs, name='train_model')
        return model