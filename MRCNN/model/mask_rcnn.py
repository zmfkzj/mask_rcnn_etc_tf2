from enum import Enum
import re

import keras.api._v2.keras as keras
import keras.api._v2.keras.layers as KL
import keras.api._v2.keras.models as KM
import numpy as np
import tensorflow as tf
import tensorflow_models as tfm

from MRCNN import utils
from MRCNN.config import Config
from MRCNN.data.dataset import Dataset
from MRCNN.metric import CocoMetric
from MRCNN.loss import MrcnnBboxLossGraph, MrcnnClassLossGraph, MrcnnMaskLossGraph, RpnBboxLossGraph, RpnClassLossGraph

from ..layer import (DetectionLayer, DetectionTargetLayer, ProposalLayer)
from ..model_utils.miscellenous_graph import NormBoxesGraph
from ..utils import compute_backbone_shapes
from . import RPN, FPN_classifier, FPN_mask, Neck


class TrainLayers(Enum):
    META = r"(meta\_.*)"
    # all layers but the backbone
    HEADS = r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(meta\_.*)"
    # All layers
    ALL = ".*"


class MaskRcnn(KM.Model):
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, config:Config, dataset:Dataset):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        super().__init__(name='mask_rcnn')
        self.config = config
        self.dataset = dataset

        if isinstance(config.GPUS, int):
            gpus = [config.GPUS]
        else:
            gpus = config.GPUS
        self.strategy = self.config.STRATEGY

        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        #parts
        self.meta_conv1 = KL.Conv2D(64,(7,7),strides=(2,2),name='meta_conv1', use_bias=True)
        self.meta_cls_score = KL.Dense(config.NUM_CLASSES, kernel_initializer=tf.initializers.HeUniform())

        self.backbone = self.make_backbone_model()
        self.neck = Neck(config)

        self.rpn = RPN(self.config.RPN_ANCHOR_STRIDE, len(self.config.RPN_ANCHOR_SCALES)*len(self.config.RPN_ANCHOR_RATIOS), name='rpn_model')

        # self.ROIAlign_classifier = PyramidROIAlign([config.POOL_SIZE, config.POOL_SIZE], self.config, name="roi_align_classifier")
        # self.ROIAlign_mask = PyramidROIAlign([config.MASK_POOL_SIZE, config.MASK_POOL_SIZE], self.config, name="roi_align_mask")

        self.ROIAlign_classifier = tfm.vision.layers.MultilevelROIAligner(config.POOL_SIZE, name="roi_align_classifier")
        self.ROIAlign_mask = tfm.vision.layers.MultilevelROIAligner(config.MASK_POOL_SIZE, name="roi_align_mask")

        self.fpn_classifier = FPN_classifier(self.config.POOL_SIZE, self.config.NUM_CLASSES, fc_layers_size=self.config.FPN_CLASSIF_FC_LAYERS_SIZE)
        self.fpn_mask = FPN_mask(self.config.NUM_CLASSES)

        self.concats = [KL.Concatenate(axis=1, name=n) for n in ["rpn_class_logits", "rpn_class", "rpn_bbox"]]
        self.anchors = self.get_anchors(self.config.IMAGE_SHAPE)

        #losses
        self.rpn_class_loss = RpnClassLossGraph(name="rpn_class_loss")
        self.rpn_bbox_loss = RpnBboxLossGraph(name="rpn_bbox_loss")
        self.class_loss = MrcnnClassLossGraph(name="mrcnn_class_loss")
        self.bbox_loss = MrcnnBboxLossGraph(name="mrcnn_bbox_loss")
        self.mask_loss = MrcnnMaskLossGraph(name="mrcnn_mask_loss")

        #excuting models
        self.predict_model = self.make_predict_model()
        self.test_model = self.make_test_model()
        self.train_model = self.make_train_model()

    
    def compile(self, coco_metric:CocoMetric, optimizer="rmsprop", loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, jit_compile=None, **kwargs):
        self.coco_metric = coco_metric
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, jit_compile, **kwargs)
    

    def predict_step(self, data):
        input_images, input_window, origin_image_shapes, pathes = data[0]
        detections, mrcnn_mask = self.predict_model([input_images, input_window])

        # detections, mrcnn_mask = self.strategy.run(self.predict_model, args=([input_images, input_window],))
        # detections = self.strategy.gather(detections,axis=0)
        # mrcnn_mask = self.strategy.gather(mrcnn_mask,axis=0)

        return detections,mrcnn_mask,origin_image_shapes, input_window, pathes
            


    def test_step(self, data):
        input_images, input_window, origin_image_shapes, image_ids = data[0]
        detections, mrcnn_mask = self.test_model([input_images, input_window])

        # detections, mrcnn_mask = self.strategy.run(self.test_model, args=([input_images, input_window],))
        # detections = self.strategy.gather(detections,axis=0)
        # mrcnn_mask = self.strategy.gather(mrcnn_mask,axis=0)

        self.coco_metric.update_state(image_ids, detections, origin_image_shapes, input_window, mrcnn_mask)
        results = self.coco_metric.result()
        mAP, mAP50, mAP75, F1_01, F1_02, F1_03, F1_04, F1_05, F1_06, F1_07, F1_08, F1_09 = tf.unstack(results)
        return {'mAP':mAP,'mAP50':mAP50,'mAP75':mAP75,'F1_0.1':F1_01,'F1_0.2':F1_02,'F1_0.3':F1_03,'F1_0.4':F1_04,'F1_0.5':F1_05,'F1_0.6':F1_06,'F1_0.7':F1_07,'F1_0.8':F1_08,'F1_0.9':F1_09}


    def train_step(self, data):
        def step_fn(resized_image, resized_boxes, minimize_masks, dataloader_class_ids,rpn_match, rpn_bbox, active_class_ids):
            with tf.GradientTape() as tape:
                rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss = \
                    self.train_model([resized_image, resized_boxes, minimize_masks, dataloader_class_ids,rpn_match, rpn_bbox, active_class_ids], training=True)
            gradients = tape.gradient(self.train_model.losses, self.train_model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.train_model.trainable_variables))
            return rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss
        
        resized_image, resized_boxes, minimize_masks, dataloader_class_ids,rpn_match, rpn_bbox, active_class_ids = data[0]
        rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss = \
            step_fn(resized_image, resized_boxes, minimize_masks, dataloader_class_ids,rpn_match, rpn_bbox, active_class_ids)

        reg_losses = self.train_model.losses[0]

        return {'rpn_class_loss':rpn_class_loss, 'rpn_bbox_loss':rpn_bbox_loss, 'class_loss':class_loss, 'bbox_loss':bbox_loss, 'mask_loss':mask_loss, 'reg_losses':reg_losses, 'lr':self.optimizer.learning_rate}

    def make_predict_model(self):
        input_image = KL.Input(self.config.IMAGE_SHAPE, dtype=tf.uint8, name='input_image')
        input_window = KL.Input(shape=(4,), name="input_window")

        backbone_output = self.backbone(input_image)
        P3,P4,P5,P6 = self.neck(*backbone_output)
        

        rpn_feature_maps = [P3, P4, P5, P6]
        mrcnn_feature_maps = {'2':P3, '3':P4, '4':P5}

        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(self.rpn(p))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        outputs = list(zip(*layer_outputs))
        outputs = [c(list(o)) for o, c in zip(outputs, self.concats)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        batch_size = tf.shape(input_image)[0]
        anchors = tf.broadcast_to(self.anchors, tf.concat([(batch_size,),tf.shape(self.anchors)],-1))
        proposal_count = self.config.POST_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(nms_threshold=self.config.RPN_NMS_THRESHOLD, name="ROI", config=self.config)([rpn_class, rpn_bbox, anchors, proposal_count])
        
        # roi_cls_feature = self.ROIAlign_classifier(rpn_rois, self.config.IMAGE_SHAPE, mrcnn_feature_maps)
        roi_cls_feature = self.ROIAlign_classifier(mrcnn_feature_maps, rpn_rois)

        # Network Heads
        # Proposal classifier and BBox regressor heads
        mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.fpn_classifier(roi_cls_feature, training=False)
        detections = DetectionLayer(self.config, name="mrcnn_detection")(rpn_rois, mrcnn_class, mrcnn_bbox, self.config.IMAGE_SHAPE, input_window)

        # Create masks for detections
        detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
        # roi_seg_feature = self.ROIAlign_mask(detection_boxes, self.config.IMAGE_SHAPE, mrcnn_feature_maps)
        roi_seg_feature = self.ROIAlign_mask(mrcnn_feature_maps, detection_boxes)
        mrcnn_mask = self.fpn_mask(roi_seg_feature, training=False)

        model = keras.Model([input_image, input_window],
                            [detections, mrcnn_mask],
                            name='mask_rcnn')
        return model
    

    def make_test_model(self):
        return self.make_predict_model()
    

    def make_train_model(self):
        input_image = KL.Input( shape=[None, None, self.config.IMAGE_SHAPE[2]], name="input_image")
        active_class_ids = KL.Input(shape=[self.config.NUM_CLASSES], name="input_class_ids")
        # RPN GT
        input_rpn_match = KL.Input( shape=[None, 1], name="input_rpn_match", dtype=tf.int64)
        input_rpn_bbox = KL.Input( shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

        # Detection GT (class IDs, bounding boxes, and masks)
        # 1. GT Class IDs (zero padded)
        input_gt_class_ids = KL.Input( shape=[None], name="input_gt_class_ids", dtype=tf.int64)
        # 2. GT Boxes in pixels (zero padded)
        # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
        input_gt_boxes = KL.Input( shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
        # Normalize coordinates
        gt_boxes = NormBoxesGraph()(input_gt_boxes, tf.shape(input_image)[1:3])
        # 3. GT Masks (zero padded)
        # [batch, height, width, MAX_GT_INSTANCES]
        input_gt_masks = KL.Input( shape=[self.config.MINI_MASK_SHAPE[0], self.config.MINI_MASK_SHAPE[1], None], name="input_gt_masks", dtype=bool)

        backbone_output = self.backbone(input_image)
        P3,P4,P5,P6 = self.neck(*backbone_output)
        

        rpn_feature_maps = [P3, P4, P5, P6]
        # mrcnn_feature_maps = [P3, P4, P5]
        mrcnn_feature_maps = {'2':P3, '3':P4, '4':P5}

        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(self.rpn(p))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        outputs = list(zip(*layer_outputs))
        outputs = [c(list(o)) for o, c in zip(outputs, self.concats)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        batch_size = tf.shape(input_image)[0]
        anchors = tf.broadcast_to(self.anchors, tf.concat([(batch_size,),tf.shape(self.anchors)],-1))
        proposal_count = self.config.POST_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(nms_threshold=self.config.RPN_NMS_THRESHOLD, name="ROI", config=self.config)([rpn_class, rpn_bbox, anchors, proposal_count])


        # Generate detection targets
        # Subsamples proposals and generates target outputs for training
        # Note that proposal class IDs, gt_boxes, and gt_masks are zero
        # padded. Equally, returned rois and targets are zero padded.
        rois, target_class_ids, target_bbox, target_mask =\
            DetectionTargetLayer(self.config, name="proposal_targets")([rpn_rois, input_gt_class_ids, gt_boxes, input_gt_masks])
        
        # roi_cls_features = self.ROIAlign_classifier(rois, self.config.IMAGE_SHAPE, mrcnn_feature_maps)
        # roi_mask_features = self.ROIAlign_mask(rois, self.config.IMAGE_SHAPE, mrcnn_feature_maps)
        roi_cls_features = self.ROIAlign_classifier(mrcnn_feature_maps, rois)
        roi_mask_features = self.ROIAlign_mask(mrcnn_feature_maps, rois)

        # Network Heads
        # TODO: verify that this handles zero padded ROIs
        mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.fpn_classifier(roi_cls_features)
        mrcnn_mask = self.fpn_mask(roi_mask_features)


        # Losses
        rpn_class_loss = self.rpn_class_loss( input_rpn_match, rpn_class_logits)
        rpn_bbox_loss = self.rpn_bbox_loss(input_rpn_bbox, input_rpn_match, rpn_bbox, batch_size)
        class_loss = self.class_loss(target_class_ids, mrcnn_class_logits, active_class_ids)
        bbox_loss = self.bbox_loss(target_bbox, target_class_ids, mrcnn_bbox)
        mask_loss = self.mask_loss(target_mask, target_class_ids, mrcnn_mask)

        # Model
        inputs = [input_image, input_gt_boxes, input_gt_masks, input_gt_class_ids, input_rpn_match, input_rpn_bbox, active_class_ids]
        outputs = [rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]

        model = keras.Model(inputs, outputs, name='mask_rcnn')

        reg_losses = [keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
                        for w in model.trainable_weights if 'gamma' not in w.name and 'beta' not in w.name]
        model.add_loss(lambda: tf.add_n(reg_losses))
        model.add_loss(outputs)
        return model


    def make_backbone_model(self):

        input = KL.Input(self.config.IMAGE_SHAPE, dtype=tf.uint8, name="input_backbone")
        x = KL.Lambda(lambda y:tf.cast(y, tf.float32), name='cast')(input)
        x = KL.Lambda(self.config.PREPROCESSING, name='preprocessing')(x)
        backbone:KM.Model = self.config.BACKBONE(input_tensor=x,include_top=False)
        x = backbone(x)

        output = []
        for i,layer in enumerate(backbone.layers[:-1]):
            if (shape:=layer.output_shape[1:3]) != backbone.layers[i+1].output_shape[1:3] \
                and len(shape)==2 \
                and tf.reduce_all(self.config.IMAGE_SHAPE[:2]%shape==0) \
                and tf.reduce_all(shape!=1):

                output.append(layer.output)
        output.append(x)
        output =output[-3:]
        # output = KL.Lambda(lambda y: y[-4:], name='pick_last_four_output')(output)

        model = keras.Model(inputs=input, outputs=output)
        
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
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        key = tuple(image_shape)
        if not key in self._anchor_cache:
            # Generate Anchors
            a = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # TODO: Remove this after the notebook are refactored to not use it
            self.anchors = a
            # Normalize coordinates
            self._anchor_cache[key] = utils.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[key]
    

    def load_weights(self, filepath):
        import h5py

        f = h5py.File(filepath, mode='r')
        saved_root_layer_names = [name if isinstance(name, str) else name.decode('utf-8') for name in f.attrs['layer_names']]
        saved_weight_names = []
        saved_weight_values = []
        for ln in saved_root_layer_names:
            for wn in f[ln].attrs['weight_names']:
                if not isinstance(wn, str):
                    wn = wn.decode('utf-8') 
                saved_weight_values.append(f[f'/{ln}/{wn}'])
                saved_weight_names.append('/'.join(wn.split('/')[-2:]))
        saved_weights = dict(zip(saved_weight_names,saved_weight_values))
        model_layers = self.collect_layers(self)
        for l in model_layers.values():
            for w in l.weights:
                weight_name = '/'.join(w.name.split('/')[-2:])
                if (weight_name in saved_weights) and (tuple(w.shape)==saved_weights[weight_name].shape):
                    w.assign(np.array(saved_weights[weight_name]))
                else:
                    print(f'{weight_name}\ can\'t assign')
        return self

    def collect_layers(self, model):
        layers = {}
        if isinstance(model, KM.Model):
            for layer in model.layers:
                layers.update(self.collect_layers(layer))
        else:
            layers[model.name]=model
        return layers