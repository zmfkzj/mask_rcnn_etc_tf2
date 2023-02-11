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


class CocoMetric(keras.metrics.Metric):
    def __init__(self, 
                 dataset:Dataset, 
                 config:Config, 
                 active_class_ids:list[int], 
                 iou_thresh: float = 0.5,
                 eval_type:EvalType=EvalType.BBOX,
                 name=None, dtype=None, **kwargs):
        super().__init__(name, dtype, **kwargs)
        self.dataset = dataset
        self.config = config
        self.active_class_ids = active_class_ids
        self.iou_thresh = iou_thresh
        self.eval_type = eval_type

        self.image_ids = []
        self.detections = []
        self.origin_image_shapes = []
        self.window = []
        self.mrcnn_mask = []


    def update_state(self, image_ids, detections,origin_image_shapes, window,mrcnn_mask=None):
        """
        Args:
            image_ids (Tensor): Tensor(shape=[batch_size]).
            detections (Tensor): Tensor(shape=[batch_size, 6]). 6 is (y1, x1, y2, x2, class_id, class_score).
            origin_image_shapes (Tensor): Tensor(shape=[batch_size, 2]). 2 is (height, width).
            window (Tensor): Tensor(shape=[batch_size, 4]). 4 is (y1, x1, y2, x2).
            mrcnn_mask (Tensor, optional): Tensor(shape=[batch_size, proposal_count, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]).
        """
        batch_size = tf.shape(image_ids)[0]
        for i in tf.range(batch_size):
            self.image_ids.append(image_ids[i])
            self.detections.append(detections[i])
            self.origin_image_shapes.append(origin_image_shapes[i])
            self.window.append(window[i])
            if mrcnn_mask is not None:
                self.mrcnn_mask.append(mrcnn_mask[i])
    

    def result(self):
        coco = deepcopy(self.dataset.coco)
        coco_results = self.build_coco_results()
        coco_results = coco.loadRes(coco_results)

        coco_eval = COCOeval(coco, coco_results, self.eval_type.value)
        coco_eval.params.imgIds = self.image_ids
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

        return results # [mAP, mAP50, mAP75, F1_0.1, F1_0.2, F1_0.3, F1_0.4, F1_0.5, F1_0.6, F1_0.7, F1_0.8, F1_0.9]



    def merge_state(self, metrics:'CocoMetric'):
        self.detections.extend(metrics.detections)
        self.origin_image_shapes.append(metrics.origin_image_shapes)
        self.window.append(metrics.window)
        self.mrcnn_mask.append(metrics.mrcnn_mask)
    

    def build_coco_results(self):
        """Arrange resutls to match COCO specs in http://cocodataset.org/#format
        """
        # If no results, return an empty list
        if self.detections:
            return []
        
        results = []
        for i, image_id in enumerate(self.image_ids):
            final_rois, final_class_ids, final_scores, final_masks =\
                unmold_detections(self.detections[i], 
                                    self.origin_image_shapes[i], 
                                    tf.constant([self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM, 3]), 
                                    self.window[i],
                                    mrcnn_mask=self.mrcnn_mask[i] if self.mrcnn_mask else None)
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