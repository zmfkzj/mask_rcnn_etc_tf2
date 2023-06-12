import re

import keras.api._v2.keras as keras
import keras.api._v2.keras.layers as KL
import keras.api._v2.keras.models as KM
import numpy as np
import pycocotools.mask as maskUtils
import tensorflow as tf
from tensorflow.python.keras.saving.saved_model.utils import no_automatic_dependency_tracking_scope

from MRCNN import utils
from MRCNN.config import Config
from MRCNN.data.dataset import Dataset
from MRCNN.data.utils import get_anchors
from MRCNN.enums import TrainLayers
from MRCNN.layer.proposal import ProposalLayer
from MRCNN.loss import MrcnnBboxLossGraph, MrcnnClassLossGraph, MrcnnMaskLossGraph, RpnBboxLossGraph, RpnClassLossGraph
from MRCNN.metric import F1Score
from ..layer import FPN
from .rpn import RPN
from .backbones import backbones

from ..utils import compute_backbone_shapes, unmold_detections


class BaseModel(KM.Model):
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """
    def __init__(self, config:Config, dataset:Dataset, name):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        super().__init__(name=name)
        self.config = config
        self.dataset = dataset
        self.num_classes = len(self.dataset.categories)

        self.anchors = get_anchors(config)
        self.anchors = utils.norm_boxes(self.anchors, config.IMAGE_SHAPE[:2])

        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")
    
        self.backbone = self.make_backbone_model(config)
        self.backbone_output_shapes = compute_backbone_shapes(config)

        self.neck = FPN(config, self.backbone.output)
        self.rpn = RPN(config.RPN_ANCHOR_STRIDE, len(config.RPN_ANCHOR_RATIOS), name='rpn_model')

        #losses
        self.rpn_class_loss = RpnClassLossGraph(name="loss_rpn_class")
        self.rpn_bbox_loss = RpnBboxLossGraph(name="loss_rpn_bbox")
        self.class_loss = MrcnnClassLossGraph(name="loss_mrcnn_class")
        self.bbox_loss = MrcnnBboxLossGraph(name="loss_mrcnn_bbox")
        self.mask_loss = MrcnnMaskLossGraph(name="loss_mrcnn_bbox")

        # for evaluation
        with no_automatic_dependency_tracking_scope(self):
            self.detection_results = []
        self.param_image_ids = set()
    

    @tf.function
    def call(self, input_image, training=False):
        backbone_output = self.backbone(input_image, training=training)
        mrcnn_feature_maps = self.neck(backbone_output)
        
        rpn_feature_maps = [mrcnn_feature_maps['3'],
                            mrcnn_feature_maps['4'],
                            mrcnn_feature_maps['5'],
                            mrcnn_feature_maps['6'],
                            mrcnn_feature_maps['7']]

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
        return rpn_rois, mrcnn_feature_maps, rpn_class_logits, rpn_bbox
    

    def predict(self, x, batch_size=None, verbose="auto", steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False):
        self.param_image_ids.clear()
        self.detection_results.clear()

        super().predict(x, batch_size, verbose, steps, callbacks, max_queue_size, workers, use_multiprocessing)
        return self.detection_results

    def evaluate(self, x=None, y=None, batch_size=None, verbose="auto", sample_weight=None, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, return_dict=False, **kwargs):
        self.param_image_ids.clear()
        self.detection_results.clear()

        super().evaluate(x, y, batch_size, verbose, sample_weight, steps, callbacks, max_queue_size, workers, use_multiprocessing, return_dict, **kwargs)

        mAP, mAP50, mAP75, mAP85, F1_01, F1_02, F1_03, F1_04, F1_05, F1_06, F1_07, F1_08, F1_09 = \
            tf.py_function(self.get_metrics, (), 
                        (tf.float16,tf.float16,tf.float16,tf.float16,tf.float16,tf.float16,tf.float16,tf.float16,tf.float16,tf.float16,tf.float16,tf.float16,tf.float16))
        
        return {'mAP':mAP,'mAP50':mAP50,'mAP75':mAP75,'mAP85':mAP85,'F1_0.1':F1_01,'F1_0.2':F1_02,'F1_0.3':F1_03,'F1_0.4':F1_04,'F1_0.5':F1_05,'F1_0.6':F1_06,'F1_0.7':F1_07,'F1_0.8':F1_08,'F1_0.9':F1_09}

    
    def make_backbone_model(self, config: Config):
        model_name = config.BACKBONE
        _, builder = backbones[model_name]
        model = builder(model_name)
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


    def get_metrics(self, eval_type='bbox', iou_thresh=0.5):
        if self.detection_results:
            results, evalImgs = self.dataset.evaluate(self.detection_results, self.param_image_ids, iou_threshold=[None,0.5,0.75,0.85], eval_type=eval_type)

            true = []
            pred = []
            sample_weight = []

            try:
                iou_idx = {iou:idx for iou,idx in zip(np.arange(0.5,1,0.05), range(10))}[float(iou_thresh)]
            except KeyError:
                raise KeyError(f'iou_thresh:{iou_thresh} must meet start(0.5),stop(1.0),step(0.05) conditions')

            for img in evalImgs:
                if img is not None and img['aRng']==[0, 10000000000.0]:
                    gtIds:list = img['gtIds']
                    dtScores = img['dtScores']
                    dtMatches = img['dtMatches'][iou_idx]
                    loader_class_id = self.dataset.get_loader_class_id(img['category_id'])

                    _true, _pred, _sample_weight = zip(*([(loader_class_id,1.,0.) if gtId in dtMatches else (loader_class_id,0.,1.) for gtId in gtIds] 
                                                        + [(0,score,1.) if gtId==0 else (loader_class_id,score,1.) for gtId, score in zip(dtMatches,dtScores)]))
                    _true = tf.one_hot(_true, len(self.dataset.categories)).numpy()
                    _pred = _true*np.expand_dims(_pred,-1)
                    true.extend(_true)
                    pred.extend(_pred)
                    sample_weight.extend(_sample_weight)


            metrics = []
            for conf_thresh in np.arange(0.1,1,0.1):
                conf_thresh = np.around(conf_thresh,1)
                metrics.append(F1Score(num_classes=len(self.dataset.categories), average='macro',threshold=conf_thresh))
            for  metric_fn in metrics:
                if true:
                    metric_fn.reset_state()
                    metric_fn.update_state(true, pred, sample_weight)
                    results.append(tf.cast(metric_fn.result(), tf.float16))
                else:
                    results.append(0.)
            
        else:
            results = [0.]*13

        return results # [mAP, mAP50, mAP75, mAP85, F1_0.1, F1_0.2, F1_0.3, F1_0.4, F1_0.5, F1_0.6, F1_0.7, F1_0.8, F1_0.9]


    def build_coco_results(self, image_ids, detections, origin_image_shapes, window, mrcnn_mask=None):
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
        if mrcnn_mask is not None:
            mrcnn_mask = mrcnn_mask.numpy()

        for b in range(image_ids.shape[0]):
            if image_ids[b]==0:
                continue

            self.param_image_ids.add(image_ids[b])

            final_rois, final_class_ids, final_scores, final_masks =\
                unmold_detections(detections[b], 
                                    origin_image_shapes[b], 
                                    self.config.IMAGE_SHAPE, 
                                    window[b],
                                    mrcnn_mask=mrcnn_mask[b] if mrcnn_mask is not None else None)

            # Loop through detections
            for j in range(final_rois.shape[0]):
                loader_class_id = final_class_ids[j]
                score = final_scores[j]
                bbox = final_rois[j]
                mask = final_masks[:, :, j] if final_masks is not None else None


                result = {
                    "image_id": image_ids[b],
                    "category_id": self.dataset.get_cat_id(loader_class_id),
                    "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                    "score": score,
                    'segmentation': maskUtils.encode(np.asfortranarray(mask)) 
                                    if mrcnn_mask is not None else []
                    }

                self.detection_results.append(result)
        

    def build_detection_results(self, pathes, detections, origin_image_shapes, window, mrcnn_mask=None):
        """
        Args:
            image_ids (Tensor): Tensor(shape=[batch_size]).
            detections (Tensor): Tensor(shape=[batch_size, detection_max_instances, 6]). 6 is (y1, x1, y2, x2, class_id, class_score).
            origin_image_shapes (Tensor): Tensor(shape=[batch_size, 2]). 2 is (height, width).
            window (Tensor): Tensor(shape=[batch_size, 4]). 4 is (y1, x1, y2, x2).
            mrcnn_mask (Tensor, optional): Tensor(shape=[batch_size, detection_max_instances, MASK_SHAPE[0], MASK_SHAPE[1], NUM_CLASSES]).
        """
        pathes = pathes.numpy()
        detections = detections.numpy()
        origin_image_shapes = origin_image_shapes.numpy()
        window = window.numpy()
        if mrcnn_mask is not None:
            mrcnn_mask = mrcnn_mask.numpy()

        for b in range(pathes.shape[0]):
            if pathes[b]==0:
                continue

            final_rois, final_class_ids, final_scores, final_masks =\
                unmold_detections(detections[b], 
                                    origin_image_shapes[b], 
                                    self.config.IMAGE_SHAPE, 
                                    window[b],
                                    mrcnn_mask=mrcnn_mask[b] if mrcnn_mask is not None else None)

            # Loop through detections
            for j in range(final_rois.shape[0]):
                loader_class_id = final_class_ids[j]
                score = final_scores[j]
                bbox = final_rois[j]
                mask = final_masks[:, :, j] if final_masks is not None else None


                result = {
                    "path": pathes[b],
                    "category_name": self.dataset.get_cat_name(self.dataset.get_cat_id(loader_class_id)),
                    "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                    "score": score,
                    'segmentation': maskUtils.encode(np.asfortranarray(mask)) 
                                    if mrcnn_mask is not None else []
                    }

                self.detection_results.append(result)