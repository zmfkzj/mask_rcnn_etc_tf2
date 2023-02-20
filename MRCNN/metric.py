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


        with tf.device('CPU'):
            self.current_count:tf.Variable = self.add_weight('current_count', initializer='zeros', dtype=tf.int32)
            size = len(self.dataset)*config.DETECTION_MAX_INSTANCES

            self.image_id:tf.Variable = self.add_weight( 
                name=f'image_ids', 
                shape=[size], 
                aggregation=tf.VariableAggregation.NONE, 
                dtype=tf.int32, 
                initializer='zeros')
            self.category_id:tf.Variable = self.add_weight( 
                name=f'category_ids', 
                shape=[size], 
                aggregation=tf.VariableAggregation.NONE, 
                dtype=tf.int32, 
                initializer='zeros')
            self.bbox:tf.Variable = self.add_weight( 
                name=f'bboxes', 
                shape=[size, 4], 
                aggregation=tf.VariableAggregation.NONE, 
                dtype=tf.float32, 
                initializer='zeros')
            self.score:tf.Variable = self.add_weight( 
                name=f'scores', 
                shape=[size], 
                aggregation=tf.VariableAggregation.NONE, 
                dtype=tf.float32, 
                initializer='zeros')
            self.segmentation:tf.Variable = self.add_weight( 
                name=f'segmentations', 
                shape=[size], 
                aggregation=tf.VariableAggregation.NONE, 
                dtype=tf.string, 
                initializer=tf.initializers.Constant(''))


    def update_state(self, image_ids, detections,origin_image_shapes, window,mrcnn_mask=None):
        """
        Args:
            image_ids (Tensor): Tensor(shape=[batch_size]).
            detections (Tensor): Tensor(shape=[batch_size, detection_max_instances, 6]). 6 is (y1, x1, y2, x2, class_id, class_score).
            origin_image_shapes (Tensor): Tensor(shape=[batch_size, 2]). 2 is (height, width).
            window (Tensor): Tensor(shape=[batch_size, 4]). 4 is (y1, x1, y2, x2).
            mrcnn_mask (Tensor, optional): Tensor(shape=[batch_size, detection_max_instances, MASK_SHAPE[0], MMASK_SHAPE[1], NUM_CLASSES]).
        """
        coco_results = tf.py_function(self.build_coco_results, 
                                      (image_ids, detections, origin_image_shapes, window, mrcnn_mask),
                                      (tf.int32, tf.int32, tf.float32, tf.float32, tf.string))
        image_id, category_id, bbox, score, segmentation = coco_results

        update_count = tf.shape(image_id)[0]

        if update_count != 0:
            self.image_id.assign(
                self.image_id.scatter_update(tf.IndexedSlices(image_id, tf.range(self.current_count, self.current_count+update_count))))
            self.category_id.assign(
                self.category_id.scatter_update(tf.IndexedSlices(category_id, tf.range(self.current_count, self.current_count+update_count))))
            self.bbox.assign(
                self.bbox.scatter_update(tf.IndexedSlices(bbox, tf.range(self.current_count, self.current_count+update_count))))
            self.score.assign(
                self.score.scatter_update(tf.IndexedSlices(score, tf.range(self.current_count, self.current_count+update_count))))
            self.segmentation.assign(
                self.segmentation.scatter_update(tf.IndexedSlices(segmentation, tf.range(self.current_count, self.current_count+update_count))))

            self.current_count.assign_add(update_count)

    
    @tf.function
    def result(self):
        return tf.py_function(self._result,[],tf.float32)


    def _result(self):
        coco = deepcopy(self.dataset.coco)
        coco_results = []
        image_ids = {}

        for i in tf.range(self.current_count):
            result = {
                "image_id": self.image_id[i].numpy(),
                "category_id": self.category_id[i].numpy(),
                "bbox": self.bbox[i].numpy(),
                "score": self.score[i].numpy(),
                'segmentation': self.segmentation[i].numpy()
            }

            coco_results.append(result)
            image_ids.update(self.image_id[i].numpy())
        
        if coco_results:
            coco_results = coco.loadRes(coco_results)

            coco_eval = COCOeval(coco, coco_results, self.eval_type.value)
            coco_eval.params.imgIds = list(image_ids)
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
        if mrcnn_mask is not None:
            mrcnn_mask:np.ndarray = mrcnn_mask.numpy()


        results_image_id = []
        results_category_id = []
        results_bbox = []
        results_score = []
        results_segmentation = []
        for i, image_id in enumerate(image_ids):
            if image_id==0:
                continue
            final_rois, final_class_ids, final_scores, final_masks =\
                unmold_detections(detections[i], 
                                    origin_image_shapes[i], 
                                    tf.constant([self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM, 3]), 
                                    window[i],
                                    mrcnn_mask=mrcnn_mask)
            # Loop through detections
            for j in tf.range(tf.shape(final_rois)[0]):
                class_id = final_class_ids[j]
                score = final_scores[j]
                bbox = final_rois[i]
                mask = final_masks[:, :, i] if final_masks is not None else None


                results_image_id.append(image_id)
                results_category_id.append(self.dataset.get_source_class_id(class_id))
                results_bbox.append([bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]])
                results_score.append(score)
                results_segmentation.append(maskUtils.encode(np.asfortranarray(mask)) if mask is not None else "")

        return (tf.constant(results_image_id, dtype=tf.int32), 
                tf.constant(results_category_id, dtype=tf.int32),
                tf.constant(results_bbox, dtype=tf.float32),
                tf.constant(results_score, dtype=tf.float32),
                tf.constant(results_segmentation, dtype=tf.string))



    @tf.function
    def reset_state(self):
        size = len(self.dataset) * self.config.DETECTION_MAX_INSTANCES
        self.image_id.assign(tf.zeros([size], dtype=tf.int32))
        self.category_id.assign(tf.zeros([size], dtype=tf.int32))
        self.bbox.assign(tf.zeros([size, 4], dtype=tf.float32))
        self.score.assign(tf.zeros([size], dtype=tf.float32))
        self.segmentation.assign(tf.zeros([size], dtype=tf.string))