import re
from copy import deepcopy
from enum import Enum
from sklearn.preprocessing import OneHotEncoder

import keras.api._v2.keras as keras
import keras.api._v2.keras.layers as KL
import keras.api._v2.keras.models as KM
import numpy as np
import pycocotools.mask as maskUtils
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_models as tfm
from pycocotools.cocoeval import COCOeval

from MRCNN import utils
from MRCNN.config import Config
from MRCNN.data.dataset import Dataset
from MRCNN.loss import (MrcnnBboxLossGraph, MrcnnClassLossGraph,
                        MrcnnMaskLossGraph, RpnBboxLossGraph,
                        RpnClassLossGraph)
from MRCNN.metric import F1Score

from ..layer import DetectionLayer, DetectionTargetLayer, ProposalLayer
from ..model_utils.miscellenous_graph import NormBoxesGraph
from ..utils import compute_backbone_shapes, unmold_detections
from . import RPN, FPN_classifier, FPN_mask, Neck


class TrainLayers(Enum):
    META = r"(meta\_.*)"
    # all layers but the backbone
    HEADS = r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(meta\_.*)"
    # All layers
    ALL = ".*"


class EvalType(Enum):
    BBOX='bbox'
    SEGM='segm'


class MaskRcnn(KM.Model):
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, config:Config):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        super().__init__(name='mask_rcnn')
        self.config = config
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

        # for evaluation
        self.val_results = []
        self.param_image_ids = set()
    

    def compile(self, dataset:Dataset, eval_type:EvalType, active_class_ids:list[int], iou_thresh: float = 0.5, optimizer="rmsprop", loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, jit_compile=None, **kwargs):
        self.dataset = dataset
        self.eval_type = eval_type
        self.active_class_ids = active_class_ids
        self.iou_thresh = iou_thresh
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, jit_compile, **kwargs)

    
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
        gradients = tape.gradient(self.train_model.losses, self.train_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.train_model.trainable_variables))
        
        reg_losses = self.train_model.losses[0]

        return {'rpn_class_loss':rpn_class_loss, 'rpn_bbox_loss':rpn_bbox_loss, 'class_loss':class_loss, 'bbox_loss':bbox_loss, 'mask_loss':mask_loss, 'reg_losses':reg_losses, 'lr':self.optimizer.learning_rate}


    def evaluate(self, x=None, y=None, batch_size=None, verbose="auto", sample_weight=None, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, return_dict=False, **kwargs):
        self.param_image_ids.clear()
        self.val_results.clear()

        super().evaluate(x, y, batch_size, verbose, sample_weight, steps, callbacks, max_queue_size, workers, use_multiprocessing, return_dict, **kwargs)

        mAP, mAP50, mAP75, F1_01, F1_02, F1_03, F1_04, F1_05, F1_06, F1_07, F1_08, F1_09 = \
            tf.py_function(self.get_coco_metrics, (), 
                           (tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32))

        return {'mAP':mAP,'mAP50':mAP50,'mAP75':mAP75,'F1_0.1':F1_01,'F1_0.2':F1_02,'F1_0.3':F1_03,'F1_0.4':F1_04,'F1_0.5':F1_05,'F1_0.6':F1_06,'F1_0.7':F1_07,'F1_0.8':F1_08,'F1_0.9':F1_09}


    def make_predict_model(self):
        input_image = KL.Input(self.config.IMAGE_SHAPE, dtype=tf.uint8, name='predict_input_image')
        input_window = KL.Input(shape=(4,), name="predict_input_window")

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
        rpn_rois = ProposalLayer(nms_threshold=self.config.RPN_NMS_THRESHOLD, name="predict_ROI", config=self.config)([rpn_class, rpn_bbox, anchors, proposal_count])
        
        # roi_cls_feature = self.ROIAlign_classifier(rpn_rois, self.config.IMAGE_SHAPE, mrcnn_feature_maps)
        roi_cls_feature = self.ROIAlign_classifier(mrcnn_feature_maps, rpn_rois)

        # Network Heads
        # Proposal classifier and BBox regressor heads
        mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.fpn_classifier(roi_cls_feature, training=False)
        detections = DetectionLayer(self.config, name="predict_mrcnn_detection")(rpn_rois, mrcnn_class, mrcnn_bbox, self.config.IMAGE_SHAPE, input_window)

        # Create masks for detections
        detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
        # roi_seg_feature = self.ROIAlign_mask(detection_boxes, self.config.IMAGE_SHAPE, mrcnn_feature_maps)
        roi_seg_feature = self.ROIAlign_mask(mrcnn_feature_maps, detection_boxes)
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
        rpn_rois = ProposalLayer(nms_threshold=self.config.RPN_NMS_THRESHOLD, name="train_ROI", config=self.config)([rpn_class, rpn_bbox, anchors, proposal_count])


        # Generate detection targets
        # Subsamples proposals and generates target outputs for training
        # Note that proposal class IDs, gt_boxes, and gt_masks are zero
        # padded. Equally, returned rois and targets are zero padded.
        rois, target_class_ids, target_bbox, target_mask =\
            DetectionTargetLayer(self.config, name="train_proposal_targets")([rpn_rois, input_gt_class_ids, gt_boxes, input_gt_masks])
        
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

        model = keras.Model(inputs, outputs, name='train_mask_rcnn')

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
    

    def collect_layers(self, model):
        layers = {}
        if isinstance(model, KM.Model):
            for layer in model.layers:
                layers.update(self.collect_layers(layer))
        else:
            layers[model.name]=model
        return layers
    

    def build_coco_results(self, image_ids, detections, origin_image_shapes, window, mrcnn_mask):
        """
        Args:
            image_ids (Tensor): Tensor(shape=[batch_size]).
            detections (Tensor): Tensor(shape=[batch_size, detection_max_instances, 6]). 6 is (y1, x1, y2, x2, class_id, class_score).
            origin_image_shapes (Tensor): Tensor(shape=[batch_size, 2]). 2 is (height, width).
            window (Tensor): Tensor(shape=[batch_size, 4]). 4 is (y1, x1, y2, x2).
            mrcnn_mask (Tensor, optional): Tensor(shape=[batch_size, detection_max_instances, MASK_SHAPE[0], MASK_SHAPE[1], NUM_CLASSES]).
        """
        image_ids = image_ids.numpy()
        detections = detections.numpy()
        origin_image_shapes = origin_image_shapes.numpy()
        window = window.numpy()
        mrcnn_mask = mrcnn_mask.numpy()

        for b in range(image_ids.shape[0]):
            if image_ids[b]==0:
                continue

            self.param_image_ids.add(image_ids[b])

            final_rois, final_class_ids, final_scores, final_masks =\
                unmold_detections(detections[b], 
                                    origin_image_shapes[b], 
                                    tf.constant([self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM, 3]), 
                                    window[b],
                                    mrcnn_mask=mrcnn_mask[b] if mrcnn_mask is not None else None)

            # Loop through detections
            for j in range(final_rois.shape[0]):
                class_id = final_class_ids[j]
                score = final_scores[j]
                bbox = final_rois[j]
                mask = final_masks[:, :, j] if final_masks is not None else None


                result = {
                    "image_id": image_ids[b],
                    "category_id": self.dataset.get_source_class_id(class_id),
                    "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                    "score": score,
                    'segmentation': maskUtils.encode(np.asfortranarray(mask)) 
                                    if mrcnn_mask is not None else []
                    }

                self.val_results.append(result)
        

    def get_coco_metrics(self):
        coco = deepcopy(self.dataset.coco)
        val_results = list(self.val_results)
        for i, r in enumerate(val_results):
            val_results[i] = {'image_id':int(r['image_id']),
                              'category_id':int(r['category_id']),
                              'bbox':[float(b) for b in r['bbox']],
                              'score':float(r['score']),
                              'segmentation':{'size':[int(s) for s in r['segmentation']['size']],
                                              'counts':r['segmentation']['counts']}}
        if val_results:
            coco_results = coco.loadRes(val_results)

            coco_eval = COCOeval(coco, coco_results, self.eval_type.value)
            coco_eval.params.imgIds = list(self.param_image_ids)
            coco_eval.params.catIds = list(self.active_class_ids)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            mAP = coco_eval.stats[0]
            mAP50 = coco_eval.stats[1]
            mAP75 = coco_eval.stats[2]

            results = [mAP, mAP50, mAP75]


            true = []
            pred = []
            sample_weight = []
            iou_idx = {iou:idx for iou,idx in zip(np.arange(0.5,1,0.05), range(10))}[float(self.iou_thresh)]
            for img in coco_eval.evalImgs:
                if img is not None and img['aRng']==[0, 10000000000.0]:
                    gtIds:list = img['gtIds']
                    dtScores = img['dtScores']
                    dtMatches = img['dtMatches'][iou_idx]
                    cat_idx = self.dataset.get_dataloader_class_id(img['category_id'])-1

                    _true, _pred, _sample_weight = zip(*([(cat_idx,1.,0.) if gtId in dtMatches else (cat_idx,0.,1.) for gtId in gtIds] 
                                                        + [(0,score,1.) if gtId==0 else (cat_idx,score,1.) for gtId, score in zip(dtMatches,dtScores)]))
                    _true = tf.one_hot(_true, len(self.active_class_ids)).numpy()
                    _pred = _true*np.expand_dims(_pred,-1)
                    true.extend(_true)
                    pred.extend(_pred)
                    sample_weight.extend(_sample_weight)


            metrics = []
            for conf_thresh in np.arange(0.1,1,0.1):
                conf_thresh = np.around(conf_thresh,1)
                metrics.append(F1Score(num_classes=len(self.active_class_ids), average='macro',threshold=conf_thresh))
            for  metric_fn in metrics:
                if true:
                    metric_fn.reset_state()
                    metric_fn.update_state(true, pred, sample_weight)
                    results.append(metric_fn.result())
                else:
                    results.append(0.)
            
        else:
            results = [0.]*12

        return results # [mAP, mAP50, mAP75, F1_0.1, F1_0.2, F1_0.3, F1_0.4, F1_0.5, F1_0.6, F1_0.7, F1_0.8, F1_0.9]