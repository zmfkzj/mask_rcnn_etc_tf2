import copy
import keras.api._v2.keras as keras
import keras.api._v2.keras.layers as KL
import tensorflow as tf
import tensorflow_models as tfm
import numpy as np
from keras.utils import tf_utils

from MRCNN.config import Config
from MRCNN.data.meta_frcnn_data_loader import InputDatas
from MRCNN.enums import EvalType, Mode
from MRCNN.layer.prn_background import PrnBackground
from MRCNN.loss import (MetaClassLoss, MrcnnBboxLossGraph, MrcnnClassLossGraph,
                        RpnBboxLossGraph, RpnClassLossGraph)
from MRCNN.model.base_model import BaseModel
from MRCNN.model.faster_rcnn import FasterRcnn
from MRCNN.model.meta_fpn_head import FPN_classifier

from ..layer import DetectionLayer, FrcnnTarget, ProposalLayer
from ..model_utils.miscellenous_graph import DenormBoxesGraph, NormBoxesGraph
from . import RPN, meta_FPN_classifier, Neck


class Outputs(tf.experimental.ExtensionType):
    detections: tf.Tensor
    attentions: tf.Tensor
    attentions_logits: tf.Tensor
    rpn_class_loss: tf.Tensor
    rpn_bbox_loss: tf.Tensor
    class_loss: tf.Tensor
    bbox_loss: tf.Tensor
    meta_loss: tf.Tensor

    def replace_data(self, new_dict):
        args = {'detections':self.detections,
                'attentions':self.attentions, 
                'attentions_logits':self.attentions_logits, 
                'rpn_class_loss':self.rpn_class_loss, 
                'rpn_bbox_loss':self.rpn_bbox_loss, 
                'class_loss':self.class_loss, 
                'bbox_loss':self.bbox_loss, 
                'meta_loss':self.meta_loss}
        args.update(new_dict)
        return Outputs(**args)


class MetaFasterRcnn(FasterRcnn):
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, config:Config, eval_type=EvalType.BBOX):
        """
        config: A Sub-class of the Config class
        """
        # super(BaseModel, self).__init__(name='faster_rcnn') # keras.Model.__init__
        # self.build_parts(config)
        # super(FasterRcnn, self).__init__(config, EvalType.BBOX) # BaseModel.__init__
        super().__init__(config, eval_type)
        self.meta_input_conv1 = KL.Conv2D(3,(1,1), name='meta_input_conv1')
        self.attention_fc = KL.Dense(config.FPN_CLASSIF_FC_LAYERS_SIZE, name="meta_attention")
        self.attention_batchnorm = KL.BatchNormalization(name='meta_bachnorm_1')
        self.meta_cls_score = KL.Dense(config.NUM_CLASSES, name='meta_score')
        self.fpn_classifier = meta_FPN_classifier(config.POOL_SIZE, config.NUM_CLASSES, fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

        self.default_outputs = Outputs(
            detections = tf.zeros([self.config.MAX_GT_INSTANCES,6], dtype=tf.float32),
            attentions = tf.zeros([self.config.NUM_CLASSES, self.config.FPN_CLASSIF_FC_LAYERS_SIZE], tf.float32),
            attentions_logits = tf.zeros([self.config.NUM_CLASSES, self.config.FPN_CLASSIF_FC_LAYERS_SIZE], tf.float32),
            rpn_class_loss = tf.constant(0., tf.float32),
            rpn_bbox_loss = tf.constant(0., tf.float32),
            class_loss = tf.constant(0., tf.float32),
            bbox_loss = tf.constant(0., tf.float32),
            meta_loss = tf.constant(0., tf.float32)
            )


    def predict_step(self, data):
        input_data:InputDatas = data[0]
        outputs:Outputs = self(input_data, Mode.PREDICT.value)

        return outputs.detections,input_data.origin_image_shapes, input_data.input_window, input_data.pathes


    def test_step(self, data):
        input_data:InputDatas = data[0]
        outputs:Outputs = self(input_data, Mode.TEST.value)
        tf.py_function(self.build_coco_results, (input_data.image_ids, outputs.detections, input_data.origin_image_shapes, input_data.input_window), ())
        return {}


    def train_step(self, data, optim):
        input_data:InputDatas = data[0]
        with tf.GradientTape() as tape:
            outputs:Outputs = self(input_data, Mode.TRAIN.value)

            rpn_class_loss = outputs.rpn_class_loss
            rpn_bbox_loss = outputs.rpn_bbox_loss
            class_loss = outputs.class_loss
            bbox_loss = outputs.bbox_loss
            meta_loss = outputs.meta_loss

            reg_losses = tf.add_n([keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
                            for w in self.trainable_weights if 'gamma' not in w.name and 'beta' not in w.name])
            
            losses = [reg_losses, 
                    rpn_class_loss    * self.loss_weights.rpn_class_loss, 
                    rpn_bbox_loss     * self.loss_weights.rpn_bbox_loss, 
                    class_loss        * self.loss_weights.mrcnn_class_loss, 
                    bbox_loss         * self.loss_weights.mrcnn_bbox_loss,
                    meta_loss         * self.loss_weights.meta_loss]

        gradients = tape.gradient(losses, self.trainable_variables)
        optim.apply_gradients(zip(gradients, self.trainable_variables))

        mean_loss = tf.reduce_mean(losses)
        reg_losses, rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, meta_loss = losses
        return {'mean_loss':mean_loss,'rpn_class_loss':rpn_class_loss, 'rpn_bbox_loss':rpn_bbox_loss, 'class_loss':class_loss, 'bbox_loss':bbox_loss, 'meta_loss':meta_loss, 'reg_losses':reg_losses, 'lr':self.optimizer.learning_rate}


    def prn_step(self, data):
        input_data:InputDatas = data[0]
        outputs:Outputs = self(input_data, Mode.PRN.value)
        return outputs.attentions


    def custom_train_function(self, dist_inputs, optim):
        losses = self.distribute_strategy.run(self.train_step, args=(dist_inputs,optim))
        mean_losses = self.distribute_strategy.reduce(tf.distribute.ReduceOp.MEAN, losses,axis=None)
        return mean_losses


    @tf.function
    def custom_test_function(self, dist_inputs):
        output = self.distribute_strategy.run(self.test_step, args=(dist_inputs,))
        mean_output = self.distribute_strategy.reduce(tf.distribute.ReduceOp.MEAN, output,axis=None)
        return mean_output
    

    @tf.function
    def infer_attentions_function(self, dist_inputs):
        batch_attentions = self.distribute_strategy.run(self.prn_step, args=(dist_inputs,))
        attentions = self.distribute_strategy.reduce(tf.distribute.ReduceOp.MEAN, batch_attentions,axis=None)
        return attentions
    

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose="auto", callbacks=None, validation_split=0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_batch_size=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False):

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
            self.stop_training = False
            train_data = next(x)
            train_function = tf.function(self.custom_train_function).get_concrete_function(train_data,self.optimizer)

            callbacks.on_train_begin()
            for epoch in range(epochs):
                self.reset_metrics()
                callbacks.on_epoch_begin(epoch)
                pbar = keras.utils.Progbar(target=self.config.STEPS_PER_EPOCH)
                for train_step,data in enumerate(x):
                    callbacks.on_train_batch_begin(train_step)
                    train_logs = train_function(data)
                    pbar.update(train_step,train_logs.items(), finalize=False)
                    callbacks.on_train_batch_end(train_step+1, train_logs)
                    if train_step==steps_per_epoch:
                        break
                
                train_logs = tf_utils.sync_to_numpy_or_python_type(train_logs)
                if train_logs is None:
                    raise ValueError(
                        "Unexpected result of `train_function` "
                        "(Empty logs). Please use "
                        "`Model.compile(..., run_eagerly=True)`, or "
                        "`tf.config.run_functions_eagerly(True)` for more "
                        "information of where went wrong, or file a "
                        "issue/bug to `tf.keras`."
                    )
                # Override with model metrics instead of last step logs
                train_logs = self._validate_and_get_metrics_result(train_logs)
                epoch_logs = copy.copy(train_logs)
                

                cls_attentions_sum = np.zeros([self.config.NUM_CLASSES, self.config.FPN_CLASSIF_FC_LAYERS_SIZE])
                cls_attentions_cnt = 0
                for attentions_step,data in enumerate(x):
                    attentions = self.infer_attentions_function(data)
                    cls_attentions_sum += attentions.numpy()
                    cls_attentions_cnt += 1
                    if attentions_step == (self.dataset.min_class_count//self.config.PRN_BATCH_SIZE + 1):
                        break
                attentions = cls_attentions_sum/cls_attentions_cnt


                if validation_data is not None:
                    callbacks.on_test_begin()
                    self.param_image_ids.clear()
                    self.val_results.clear()
                    for test_step,data in enumerate(validation_data):
                        callbacks.on_test_batch_begin(test_step)
                        input_datas:InputDatas = data[0]
                        input_datas = input_datas.replace_data({'attentions':tf.cast(attentions, tf.float32)})

                        self.custom_test_function([input_datas])

                        callbacks.on_test_batch_end(test_step+1)

                        if test_step==validation_steps:
                            break

                    mAP, mAP50, mAP75, F1_01, F1_02, F1_03, F1_04, F1_05, F1_06, F1_07, F1_08, F1_09 = self.get_coco_metrics()
                    test_logs = {'mAP':mAP,'mAP50':mAP50,'mAP75':mAP75,'F1_0.1':F1_01,'F1_0.2':F1_02,'F1_0.3':F1_03,'F1_0.4':F1_04,'F1_0.5':F1_05,'F1_0.6':F1_06,'F1_0.7':F1_07,'F1_0.8':F1_08,'F1_0.9':F1_09}
                    test_logs = tf_utils.sync_to_numpy_or_python_type(test_logs)
                    # Override with model metrics instead of last step logs
                    test_logs = self._validate_and_get_metrics_result(test_logs)
                    test_logs = { "val_" + name: val for name, val in test_logs.items() }

                    callbacks.on_test_end(logs=test_logs)
                    epoch_logs.update(test_logs)
                callbacks.on_epoch_end(epoch, epoch_logs)
                if self.stop_training:
                    break
                pbar.update(train_step,epoch_logs.items(), finalize=True)
            callbacks.on_train_end(logs=epoch_logs)


    def call(self, input_datas:InputDatas, mode:str) -> Outputs:
        if mode in [Mode.PREDICT.value, Mode.TEST.value]:
            outputs:Outputs = self.forward_predict_test(input_datas.input_images,
                                                input_datas.input_window,
                                                input_datas.attentions)
        elif mode==Mode.TRAIN.value:
            outputs:Outputs = self.forward_timedist_prn(input_datas.input_images, 
                                               input_datas.input_gt_boxes,
                                               input_datas.prn_images)
            attention_score = self.meta_cls_score(outputs.attentions_logits)
            meta_loss = MetaClassLoss(name='meta_class_loss')(attention_score)

            outputs:Outputs = self.forward_train(input_datas.input_images,
                                         input_datas.active_class_ids,
                                         input_datas.rpn_match,
                                         input_datas.rpn_bbox,
                                         input_datas.dataloader_class_ids,
                                         input_datas.input_gt_boxes,
                                         outputs.attentions)
            outputs:Outputs = outputs.replace_data({'meta_loss':meta_loss})
        elif mode == Mode.PRN.value:
            outputs = self.forward_timedist_prn(input_datas.input_images, 
                                               input_datas.input_gt_boxes,
                                               input_datas.prn_images)
        else:
            outputs = Outputs()
            ValueError('argument mode must be one of predict, test, train, or prn.')
        return outputs


    @tf.function
    def forward_predict_test(self, input_image, input_window, input_attentions):
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
        rpn_outputs = list(zip(*layer_outputs))
        rpn_outputs = [c(list(o)) for o, c in zip(rpn_outputs, [KL.Concatenate(axis=1, name=n) for n in ["rpn_class_logits", "rpn_class", "rpn_bbox"]])]

        rpn_class_logits, rpn_class, rpn_bbox = rpn_outputs

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
        detections = DetectionLayer(self.config, name="detection")(rpn_rois, mrcnn_class, mrcnn_bbox, self.config.IMAGE_SHAPE, tf.cast(input_window, tf.float32))

        outputs = self.default_outputs.replace_data({'detections':detections})
        return outputs

    
    @tf.function
    def forward_train(self, input_image, active_class_ids, input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, attentions):
        # Normalize coordinates
        gt_boxes = NormBoxesGraph()(tf.cast(input_gt_boxes,tf.float32), tf.shape(input_image)[1:3])

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
        rpn_outputs = list(zip(*layer_outputs))
        rpn_outputs = [c(list(o)) for o, c in zip(rpn_outputs, [KL.Concatenate(axis=1, name=n) for n in ["rpn_class_logits", "rpn_class", "rpn_bbox"]])]

        rpn_class_logits, rpn_class, rpn_bbox = rpn_outputs

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

        outputs = self.default_outputs.replace_data({
            'rpn_class_loss':rpn_class_loss, 
            'rpn_bbox_loss':rpn_bbox_loss, 
            'class_loss':class_loss, 
            'bbox_loss':bbox_loss})

        return outputs


    @tf.function
    def forward_timedist_prn(self, input_image, input_gt_boxes, batch_prn_images):
        prn_images = PrnBackground(self.config)([input_image, input_gt_boxes, batch_prn_images])
        batch_attentions_logits, batch_attentions = tf.vectorized_map(self.forward_prn,prn_images)
        attentions_logits = tf.reduce_mean(batch_attentions_logits, 0) # [NUM_CLASSES, FC_LAYERS_SIZE]
        attentions = tf.reduce_mean(batch_attentions, 0) # [NUM_CLASSES, FC_LAYERS_SIZE]
        output = self.default_outputs.replace_data({'attentions':attentions, 'attentions_logits':attentions_logits})
        return output


    @tf.function
    def forward_prn(self, input_prn_image):
        x = self.meta_input_conv1(input_prn_image)
        backbone_output = self.backbone(x)
        prn_P2, prn_P3, prn_P4, prn_P5, prn_P6 = self.neck(*backbone_output)

        Gavg_P2 = KL.GlobalAveragePooling2D()(prn_P2)
        Gavg_P3 = KL.GlobalAveragePooling2D()(prn_P3)
        Gavg_P4 = KL.GlobalAveragePooling2D()(prn_P4)
        Gavg_P5 = KL.GlobalAveragePooling2D()(prn_P5)

        attentions = tf.reduce_mean(tf.stack([Gavg_P2, Gavg_P3, Gavg_P4, Gavg_P5]),0)
        attentions_logits = self.attention_fc(attentions)
        attentions = KL.Activation(tf.nn.sigmoid)(attentions_logits)

        return attentions_logits,attentions