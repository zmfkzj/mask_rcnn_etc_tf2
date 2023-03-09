import copy
import keras.api._v2.keras as keras
import keras.api._v2.keras.layers as KL
import tensorflow as tf
import tensorflow_models as tfm
import numpy as np
from keras.utils import tf_utils

from MRCNN.config import Config
from MRCNN.enums import EvalType
from MRCNN.layer.prn_background import PrnBackground
from MRCNN.loss import (MetaClassLoss, MrcnnBboxLossGraph, MrcnnClassLossGraph,
                        RpnBboxLossGraph, RpnClassLossGraph)
from MRCNN.model.base_model import BaseModel
from MRCNN.model.faster_rcnn import FasterRcnn

from ..layer import DetectionLayer, FrcnnTarget, ProposalLayer
from ..model_utils.miscellenous_graph import DenormBoxesGraph, NormBoxesGraph
from . import RPN, meta_FPN_classifier, Neck


class MetaFasterRcnn(FasterRcnn):
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """
    def __init__(self, config:Config):
        """
        config: A Sub-class of the Config class
        """
        super(BaseModel, self).__init__(name='faster_rcnn') # keras.Model.__init__
        self.build_parts(config)
        super(FasterRcnn, self).__init__(config, EvalType.BBOX) # BaseModel.__init__

    
    def build_parts(self, config: Config):
        super().build_parts(config)
        #additional parts
        self.neck = Neck(config)
        self.rpn = RPN(config.RPN_ANCHOR_STRIDE, len(config.RPN_ANCHOR_RATIOS), name='rpn_model')

        self.ROIAlign_classifier = tfm.vision.layers.MultilevelROIAligner(config.POOL_SIZE, name="roi_align_classifier")
        self.fpn_classifier = meta_FPN_classifier(config.POOL_SIZE, config.NUM_CLASSES, fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

        #meta parts
        self.meta_input_conv1 = KL.Conv2D(3,(1,1), name='meta_input_conv1')
        self.attention_fc = KL.Dense(config.FPN_CLASSIF_FC_LAYERS_SIZE, name="meta_attention")
        self.attention_batchnorm = KL.BatchNormalization(name='meta_bachnorm_1')
        self.meta_cls_score = KL.Dense(config.NUM_CLASSES, name='meta_score')
        self.prn_model:keras.Model = KL.TimeDistributed(self.make_prn_model(config))


    def predict_step(self, data):
        input_images, input_window, origin_image_shapes, pathes, attentions = data[0]
        detections = self.predict_model([input_images, input_window, attentions])
        return detections,origin_image_shapes, input_window, pathes


    def test_step(self, data):
        input_images, input_window, origin_image_shapes, image_ids, attentions = data[0]
        detections = self.test_model([input_images, input_window, attentions])
        tf.py_function(self.build_coco_results, (image_ids, detections, origin_image_shapes, input_window), ())
        return {}


    def train_step(self, data):
        resized_image, resized_boxes, dataloader_class_ids,rpn_match, rpn_bbox, active_class_ids, prn_images = data[0]
        with tf.GradientTape() as tape:
            rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, meta_loss = \
                self.train_model([resized_image, resized_boxes, dataloader_class_ids,rpn_match, rpn_bbox, active_class_ids, prn_images], training=True)

            reg_losses = tf.add_n([keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
                            for w in self.train_model.trainable_weights if 'gamma' not in w.name and 'beta' not in w.name])
            
            losses = [reg_losses, 
                    rpn_class_loss    * self.loss_weights.rpn_class_loss, 
                    rpn_bbox_loss     * self.loss_weights.rpn_bbox_loss, 
                    class_loss        * self.loss_weights.mrcnn_class_loss, 
                    bbox_loss         * self.loss_weights.mrcnn_bbox_loss,
                    meta_loss         * self.loss_weights.meta_loss]

        gradients = tape.gradient(losses, self.train_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.train_model.trainable_variables))

        # losses = [loss/self.config.GPU_COUNT for loss in losses]
        mean_loss = tf.reduce_mean(losses)
        reg_losses, rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, meta_loss = losses
        return {'mean_loss':mean_loss,'rpn_class_loss':rpn_class_loss, 'rpn_bbox_loss':rpn_bbox_loss, 'class_loss':class_loss, 'bbox_loss':bbox_loss, 'meta_loss':meta_loss, 'reg_losses':reg_losses, 'lr':self.optimizer.learning_rate}

    
    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose="auto", callbacks=None, validation_split=0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_batch_size=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False):
        def train_function(dist_inputs):
            losses = self.distribute_strategy.run(self.train_step, args=(dist_inputs,))
            mean_losses = self.distribute_strategy.reduce(tf.distribute.ReduceOp.MEAN, losses,axis=None)
            return mean_losses

        def test_function(dist_inputs):
            output = self.distribute_strategy.run(self.test_step, args=(dist_inputs,))
            mean_output = self.distribute_strategy.reduce(tf.distribute.ReduceOp.MEAN, output,axis=None)
            return mean_output


        with self.distribute_strategy.scope():
            if not isinstance(callbacks, keras.callbacks.CallbackList):
                callbacks = keras.callbacks.CallbackList(
                    callbacks,
                    add_history=True,
                    add_progbar=verbose != 0,
                    model=self,
                    verbose=verbose,
                    epochs=epochs,
                )

            callbacks.on_train_begin()
            for epoch in range(epochs):
                self.reset_metrics()
                callbacks.on_epoch_begin(epoch)
                for step,data in enumerate(x):
                    callbacks.on_train_batch_begin(step)
                    train_logs = train_function(data)
                    callbacks.on_train_batch_end(step+1, train_logs)
                    if step==steps_per_epoch:
                        break
                
                logs = tf_utils.sync_to_numpy_or_python_type(logs)
                if logs is None:
                    raise ValueError(
                        "Unexpected result of `train_function` "
                        "(Empty logs). Please use "
                        "`Model.compile(..., run_eagerly=True)`, or "
                        "`tf.config.run_functions_eagerly(True)` for more "
                        "information of where went wrong, or file a "
                        "issue/bug to `tf.keras`."
                    )
                # Override with model metrics instead of last step logs
                logs = self._validate_and_get_metrics_result(logs)
                epoch_logs = copy.copy(logs)
                

                cls_attentions_sum = np.zeros([self.config.NUM_CLASSES, self.config.FPN_CLASSIF_FC_LAYERS_SIZE])
                cls_attentions_cnt = 0
                for step,data in enumerate(x):
                    resized_image, resized_boxes, dataloader_class_ids,rpn_match, rpn_bbox, active_class_ids, prn_images = data[0]
                    batch_attentions = self.prn_model.predict(prn_images)
                    cls_attentions_sum += tf.reduce_mean(batch_attentions,0).numpy()
                    cls_attentions_cnt += 1
                    batch_size = resized_image.shape[0]
                    if step == (self.dataset.min_class_count//batch_size + 1):
                        break
                attentions = cls_attentions_sum/cls_attentions_cnt
                attentions = tf.broadcast_to(tf.expand_dims(attentions,0), [batch_size, *attentions.shape])


                if validation_data is not None:
                    callbacks.on_test_begin()
                    for step,data in enumerate(validation_data):
                        callbacks.on_test_batch_begin(step)
                        input_images, input_window, origin_image_shapes, image_ids = data[0]
                        test_logs = test_function([[input_images, input_window, origin_image_shapes, image_ids, attentions]])
                        callbacks.on_test_batch_end(step+1, test_logs)

                        if step==validation_steps:
                            break
                    test_logs = tf_utils.sync_to_numpy_or_python_type(test_logs)
                    # Override with model metrics instead of last step logs
                    test_logs = self._validate_and_get_metrics_result(test_logs)
                    test_logs = { "val_" + name: val for name, val in test_logs.items() }

                    callbacks.on_test_end(logs=test_logs)
                    epoch_logs.update(test_logs)
                callbacks.on_epoch_end(epoch, epoch_logs)
            callbacks.on_train_end(logs=epoch_logs)


    def make_predict_model(self):
        input_image = KL.Input(self.config.IMAGE_SHAPE, dtype=tf.uint8, name='predict_input_image')
        input_window = KL.Input(shape=(4,), name="predict_input_window")
        input_attentions = KL.Input( shape=[None, self.config.NUM_CLASSES, self.config.FPN_CLASSIF_FC_LAYERS_SIZE], name="input_attentions", dtype=tf.float32)

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
        mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.fpn_classifier(roi_cls_feature, input_attentions, training=False)
        detections = DetectionLayer(self.config, name="detection")(rpn_rois, mrcnn_class, mrcnn_bbox, self.config.IMAGE_SHAPE, input_window)

        model = keras.Model([input_image, input_window, input_attentions],
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

        input_prn_images = KL.Input(shape=[self.config.NUM_CLASSES-1, *self.config.PRN_IMAGE_SIZE, 4], name="train_input_prn_images", dtype=tf.float32)

        prn_images = PrnBackground(self.config)(input_image, input_gt_boxes, input_prn_images)
        attentions = self.prn_model(prn_images)
        attentions = tf.reduce_mean(attentions, 0) # [NUM_CLASSES, FC_LAYERS_SIZE]
        attention_score = self.meta_cls_score(attentions)

        backbone_output = self.backbone(input_image)
        P2,P3,P4,P5,P6 = self.neck(*backbone_output) # [batch_size, None, None, TOP_DOWN_PYRAMID_SIZE]

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
        mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.fpn_classifier(roi_cls_features,attentions)


        # Losses
        rpn_class_loss = RpnClassLossGraph(name="rpn_class_loss")( input_rpn_match, rpn_class_logits)
        rpn_bbox_loss = RpnBboxLossGraph(name="rpn_bbox_loss")(input_rpn_bbox, input_rpn_match, rpn_bbox, batch_size)
        class_loss = MrcnnClassLossGraph(name="mrcnn_class_loss")(target_class_ids, mrcnn_class_logits, active_class_ids)
        bbox_loss = MrcnnBboxLossGraph(name="mrcnn_bbox_loss")(target_bbox, target_class_ids, mrcnn_bbox)
        meta_loss = MetaClassLoss(name='meta_class_loss')(attention_score)

        # Model
        inputs = [input_image, input_gt_boxes, input_gt_class_ids, input_rpn_match, input_rpn_bbox, active_class_ids, input_prn_images]
        outputs = [rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, meta_loss]

        model = keras.Model(inputs, outputs, name='train_model')
        return model


    def make_prn_model(self, config: Config):
        input_image = KL.Input(config.PRN_IMAGE_SIZE+(4,), name='prn_input', dtype=tf.float32)
        x = self.meta_input_conv1(input_image)
        backbone_output = self.backbone(x)
        prn_P2, prn_P3, prn_P4, prn_P5, prn_P6 = self.neck(*backbone_output)

        Gavg_P2 = KL.GlobalAveragePooling2D()(prn_P2)
        Gavg_P3 = KL.GlobalAveragePooling2D()(prn_P3)
        Gavg_P4 = KL.GlobalAveragePooling2D()(prn_P4)
        Gavg_P5 = KL.GlobalAveragePooling2D()(prn_P5)

        attentions = tf.reduce_mean(tf.stack([Gavg_P2, Gavg_P3, Gavg_P4, Gavg_P5]),0)
        attentions = self.attention_fc(attentions)
        attentions = self.attention_batchnorm(attentions)
        attentions = KL.Activation(tf.nn.sigmoid)(attentions)

        model = keras.Model(inputs=input_image, outputs=attentions, name='prn')
        return model
        
