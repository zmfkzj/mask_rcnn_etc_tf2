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
from MRCNN.data.dataset import Dataset
from MRCNN.data.input_datas import InputDatas
from MRCNN.enums import EvalType, Mode, TrainLayers
from MRCNN.metric import F1Score
from MRCNN.model.outputs import Outputs

from ..utils import LossWeight, compute_backbone_shapes, unmold_detections


class BaseModel(KM.Model):
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """
    def __init__(self, config:Config, eval_type:EvalType, name):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        super().__init__(name=name)
        self.config = config
        self.strategy = self.config.STRATEGY
        self.eval_type = eval_type

        self.anchors = self.get_anchors(config.IMAGE_SHAPE)

        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")
        # for evaluation
        with no_automatic_dependency_tracking_scope(self):
            self.val_results = []
        self.param_image_ids = set()
    
        self.backbone = self.make_backbone_model(config)
    
    
    def compile(self, dataset:Dataset, 
                active_class_ids:Optional[list[int]], 
                iou_thresh: float = 0.5, 
                optimizer="rmsprop", 
                train_layers=TrainLayers.ALL,
                loss_weights:LossWeight=LossWeight(), 
                loss=None, 
                metrics=None, 
                weighted_metrics=None, 
                run_eagerly=None, 
                steps_per_execution=None, 
                jit_compile=None, 
                **kwargs):
        self.dataset = dataset
        self.active_class_ids = active_class_ids
        self.iou_thresh = iou_thresh
        self.loss_weights = loss_weights
        dummy_data = InputDatas(self.config, self.anchors.shape[0]).to_dict()
        self.set_call_function(Mode.TRAIN)
        self(dummy_data)
        self.set_trainable(train_layers)
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, jit_compile, **kwargs)
    

    def set_call_function(self, mode:Mode):
        self.call_function = None

    
    def call(self, inputs) -> Outputs:
        input_datas:InputDatas = InputDatas(self.config, **inputs)
        return self.call_function(input_datas)

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose="auto", callbacks=None, validation_split=0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_batch_size=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False):
        self.set_call_function(Mode.TRAIN)
        return super().fit(x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)
    
    def predict(self, x, batch_size=None, verbose="auto", steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False):
        self.set_call_function(Mode.PREDICT)

        self.param_image_ids.clear()
        self.val_results.clear()

        super().predict(x, batch_size, verbose, steps, callbacks, max_queue_size, workers, use_multiprocessing)
        return self.val_results

    def evaluate(self, x=None, y=None, batch_size=None, verbose="auto", sample_weight=None, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, return_dict=False, **kwargs):
        self.param_image_ids.clear()
        self.val_results.clear()

        self.set_call_function(Mode.TEST)
        super().evaluate(x, y, batch_size, verbose, sample_weight, steps, callbacks, max_queue_size, workers, use_multiprocessing, return_dict, **kwargs)

        mAP, mAP50, mAP75, F1_01, F1_02, F1_03, F1_04, F1_05, F1_06, F1_07, F1_08, F1_09 = \
            tf.py_function(self.get_custom_metrics, (), 
                        (tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32))
        
        return {'mAP':mAP,'mAP50':mAP50,'mAP75':mAP75,'F1_0.1':F1_01,'F1_0.2':F1_02,'F1_0.3':F1_03,'F1_0.4':F1_04,'F1_0.5':F1_05,'F1_0.6':F1_06,'F1_0.7':F1_07,'F1_0.8':F1_08,'F1_0.9':F1_09}

    
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
    

    def collect_layers(self, model):
        layers = {}
        if isinstance(model, KM.Model):
            for layer in model.layers:
                layers.update(self.collect_layers(layer))
        else:
            layers[model.name]=model
        return layers
    

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
        

    def get_coco_eval(self):
        coco = deepcopy(self.dataset.coco)
        val_results = copy(self.val_results)
        self.val_results.clear()

        with Manager() as mng:
            return_value = mng.list()
            p = Process(target=self.__get_coco_eval, args=(coco, val_results, self.eval_type, self.param_image_ids, self.active_class_ids, return_value))
            p.start()
            p.join()
            r = list(return_value)
        return r
    

    @staticmethod
    def __get_coco_eval(coco,val_results, eval_type, param_image_ids, active_class_ids, return_value):
        coco_results = coco.loadRes(val_results)

        coco_eval = COCOeval(coco, coco_results, eval_type.value)
        coco_eval.params.imgIds = list(param_image_ids)
        coco_eval.params.catIds = list(active_class_ids)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        mAP = coco_eval.stats[0]
        mAP50 = coco_eval.stats[1]
        mAP75 = coco_eval.stats[2]


        return_value.append(mAP)
        return_value.append(mAP50)
        return_value.append(mAP75)
        return_value.append(coco_eval.evalImgs)
        

    def get_custom_metrics(self):
        val_results = self.val_results
        if val_results:
            mAP, mAP50, mAP75, evalImgs = self.get_coco_eval()
            results = [mAP, mAP50, mAP75]


            true = []
            pred = []
            sample_weight = []
            iou_idx = {iou:idx for iou,idx in zip(np.arange(0.5,1,0.05), range(10))}[float(self.iou_thresh)]
            for img in evalImgs:
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
                class_id = final_class_ids[j]
                score = final_scores[j]
                bbox = final_rois[j]
                mask = final_masks[:, :, j] if final_masks is not None else None


                result = {
                    "path": pathes[b],
                    "category_id": class_id,
                    "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                    "score": score,
                    'segmentation': maskUtils.encode(np.asfortranarray(mask)) 
                                    if mrcnn_mask is not None else []
                    }

                self.val_results.append(result)