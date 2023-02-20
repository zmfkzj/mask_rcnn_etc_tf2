from collections import defaultdict
from copy import deepcopy
from enum import Enum

import keras.api._v2.keras as keras
import numpy as np
import pycocotools.mask as maskUtils
import tensorflow as tf
import tensorflow_addons as tfa
from pycocotools.cocoeval import COCOeval

from MRCNN.config import Config
from MRCNN.data.dataset import Dataset
from MRCNN.utils import unmold_detections


class EvalType(Enum):
    BBOX='bbox'
    SEGM='segm'

keras.metrics.Accuracy


class CocoMetric(keras.metrics.Metric):
    def __init__(self, 
                 dataset:Dataset, 
                 config:Config, 
                 active_class_ids:list[int], 
                 iou_thresh: float = 0.5,
                 eval_type:EvalType=EvalType.BBOX,
                 include_mask:bool=True,
                 name=None, dtype=None, **kwargs):
        super().__init__(name, dtype, **kwargs)
        self.dataset = dataset
        self.config = config
        self.active_class_ids = active_class_ids
        self.iou_thresh = iou_thresh
        self.eval_type = eval_type
        self.include_mask = include_mask


        self.image_ids:tf.Variable = self.add_weight(
            name='image_ids', 
            shape=[len(self.dataset)], 
            aggregation=tf.VariableAggregation.NONE, 
            dtype=tf.int32, 
            initializer='zeros')
        self.detections:tf.Variable = self.add_weight(
            name='detections', 
            shape=[len(self.dataset), config.DETECTION_MAX_INSTANCES,6], 
            aggregation=tf.VariableAggregation.NONE, 
            dtype=tf.float32, 
            initializer='zeros')
        self.origin_image_shapes:tf.Variable = self.add_weight(
            name='origin_image_shapes', 
            shape=[len(self.dataset),2], 
            aggregation=tf.VariableAggregation.NONE, 
            dtype=tf.int32, 
            initializer='zeros')
        self.windows:tf.Variable = self.add_weight(
            name='windows', 
            shape=[len(self.dataset),4], 
            aggregation=tf.VariableAggregation.NONE, 
            dtype=tf.int32, 
            initializer='zeros')
        if include_mask:
            self.mrcnn_mask:tf.Variable = self.add_weight(
                name='mrcnn_mask', 
                shape=[len(self.dataset),config.DETECTION_MAX_INSTANCES,*config.MASK_SHAPE,config.NUM_CLASSES], 
                aggregation=tf.VariableAggregation.NONE, 
                dtype=tf.float32, 
                initializer='zeros')


    @tf.function
    def update_state(self, image_ids, detections,origin_image_shapes, window,mrcnn_mask=None):
        """
        Args:
            image_ids (Tensor): Tensor(shape=[batch_size]).
            detections (Tensor): Tensor(shape=[batch_size, detection_max_instances, 6]). 6 is (y1, x1, y2, x2, class_id, class_score).
            origin_image_shapes (Tensor): Tensor(shape=[batch_size, 2]). 2 is (height, width).
            window (Tensor): Tensor(shape=[batch_size, 4]). 4 is (y1, x1, y2, x2).
            mrcnn_mask (Tensor, optional): Tensor(shape=[batch_size, detection_max_instances, MASK_SHAPE[0], MMASK_SHAPE[1], NUM_CLASSES]).
        """
        current_count = tf.reduce_sum(tf.cast(tf.cast(image_ids, tf.bool), tf.int32))
        batch_size = tf.shape(image_ids)[0]

        self.image_ids.assign(
            self.image_ids.scatter_update(
            tf.IndexedSlices(image_ids, tf.range(current_count,current_count+batch_size))))
        self.detections.assign(
            self.detections.scatter_update(
            tf.IndexedSlices(detections, tf.range(current_count,current_count+batch_size))))
        self.origin_image_shapes.assign(
            self.origin_image_shapes.scatter_update(
            tf.IndexedSlices(origin_image_shapes, tf.range(current_count,current_count+batch_size))))
        self.windows.assign(
            self.windows.scatter_update(
            tf.IndexedSlices(window, tf.range(current_count,current_count+batch_size))))
        if self.include_mask:
            self.mrcnn_mask.assign(
                self.mrcnn_mask.scatter_update(
                tf.IndexedSlices(mrcnn_mask, tf.range(current_count,current_count+batch_size))))

    
    @tf.function
    def result(self):
        return tf.py_function(self._result,[self.image_ids],tf.float32)


    def _result(self, image_ids):
        coco = deepcopy(self.dataset.coco)
        coco_results = self.build_coco_results(self.image_ids, self.detections, self.origin_image_shapes, self.windows, self.mrcnn_mask)
        if coco_results:
            coco_results = coco.loadRes(coco_results)

            coco_eval = COCOeval(coco, coco_results, self.eval_type.value)
            coco_eval.params.imgIds = image_ids.numpy()
            coco_eval.params.catIds = self.active_class_ids
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
            iou_idx = {iou:idx for iou,idx in zip(np.arange(0.5,1,0.05), range(10))}[self.iou_thresh]
            for img in coco_eval.evalImgs:
                if img is not None and img['aRng']==[0, 10000000000.0]:
                    gtIds:list = img['gtIds']
                    dtScores = img['dtScores']
                    dtMatches = img['dtMatches'][iou_idx]
                    cat_idx = ([0]+self.active_class_ids).index(img['category_id'])

                    _true, _pred, _sample_weight = zip(*([(cat_idx,1,0) if gtId in dtMatches else (cat_idx,0,1) for gtId in gtIds] 
                                                        + [(0,score,1) if gtId==0 else (cat_idx,score,1) for gtId, score in zip(dtMatches,dtScores)]))
                    _true = tf.one_hot(_true, len(self.active_class_ids)+1)
                    _pred = _true*tf.expand_dims(tf.constant(_pred, tf.float32),-1)
                    _sample_weight = tf.constant(_sample_weight,tf.float32)
                    true.extend(_true)
                    pred.extend(_pred)
                    sample_weight.extend(_sample_weight)


            metrics = []
            for conf_thresh in tf.range(0.1,1,0.1):
                metrics.append(tfa.metrics.F1Score(num_classes=len(self.active_class_ids)+1, average=None,threshold=conf_thresh))
            for  metric_fn in metrics:
                if true:
                    metric_fn.reset_state()
                    metric_fn.update_state(true, pred, sample_weight)
                    results.append(tf.reduce_mean(metric_fn.result()[1:]))
                else:
                    results.append(0.)
            
            results = tf.stack(results)
        
        else:
            results = tf.stack([0.]*12)

        return results # [mAP, mAP50, mAP75, F1_0.1, F1_0.2, F1_0.3, F1_0.4, F1_0.5, F1_0.6, F1_0.7, F1_0.8, F1_0.9]


    def build_coco_results(self, image_ids, detections, origin_image_shapes, window, mrcnn_mask):
        """Arrange resutls to match COCO specs in http://cocodataset.org/#format
        """
        image_ids:np.ndarray = image_ids.numpy()
        detections:np.ndarray = detections.numpy()
        origin_image_shapes:np.ndarray = origin_image_shapes.numpy()
        window:np.ndarray = window.numpy()
        mrcnn_mask:np.ndarray = mrcnn_mask.numpy()

        results = []
        for i, image_id in enumerate(image_ids):
            if image_id==0:
                continue
            final_rois, final_class_ids, final_scores, final_masks =\
                unmold_detections(detections[i], 
                                    origin_image_shapes[i], 
                                    tf.constant([self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM, 3]), 
                                    window[i],
                                    mrcnn_mask=mrcnn_mask[i] if mrcnn_mask.size!=0 else None)
            # Loop through detections
            for j in tf.range(tf.shape(final_rois)[0]):
                class_id = final_class_ids[j]
                score = final_scores[j]
                bbox = final_rois[i]
                mask = final_masks[:, :, i] if final_masks is not None else None

                result = {
                    "image_id": image_id,
                    "category_id": self.dataset.get_source_class_id(class_id),
                    "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                    "score": score
                }

                if mask is not None:
                    result.update({"segmentation": maskUtils.encode(np.asfortranarray(mask))})
                else:
                    result.update({"segmentation": []})

                results.append(result)
        return results


    @tf.function
    def reset_state(self):
        self.image_ids.assign(
            tf.zeros([len(self.dataset)], 
                     dtype=tf.int32))
        self.detections.assign(
            tf.zeros([len(self.dataset), self.config.DETECTION_MAX_INSTANCES,6], 
                     dtype=tf.float32))
        self.origin_image_shapes.assign(
            tf.zeros([len(self.dataset),2], 
                     dtype=tf.int32))
        self.windows.assign(
            tf.zeros([len(self.dataset),4], 
                     dtype=tf.int32))
        self.mrcnn_mask.assign(
            tf.zeros([len(self.dataset),self.config.DETECTION_MAX_INSTANCES,*self.config.MASK_SHAPE,self.config.NUM_CLASSES], 
                     dtype=tf.float32))