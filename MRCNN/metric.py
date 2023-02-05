import tensorflow as tf
import keras.api._v2.keras as keras
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
import numpy as np
from MRCNN.config import Config

from MRCNN.utils import unmold_detections


class COCOMetric(keras.metrics.Metric):
    def __init__(self, dataset, config:Config, name=None, dtype=None, **kwargs):
        super().__init__(name, dtype, **kwargs)
        self.dataset = dataset
        self.config = config

        self.detections = []
        self.origin_image_shapes = []
        self.window = []
        self.mrcnn_mask = []


    def update_state(self, detections,origin_image_shapes, window,mrcnn_mask=None):
        self.detections.append(detections)
        self.origin_image_shapes.append(origin_image_shapes)
        self.window.append(window)
        if mrcnn_mask is not None:
            self.mrcnn_mask.append(mrcnn_mask)
    

    def result(self):
        coco_results = self.build_coco_results()


    def merge_state(self, metrics:'COCOMetric'):
        self.detections.extend(metrics.detections)
        self.origin_image_shapes.append(metrics.origin_image_shapes)
        self.window.append(metrics.window)
        self.mrcnn_mask.append(metrics.mrcnn_mask)
    

    def build_coco_results(self, image_ids, rois, class_ids, scores, masks):
        """Arrange resutls to match COCO specs in http://cocodataset.org/#format
        """
        # If no results, return an empty list
        if rois is None:
            return []

        results = []
        for image_id in image_ids:
            # Loop through detections
            for i in range(rois.shape[0]):
                class_id = class_ids[i]
                score = scores[i]
                bbox = np.around(rois[i], 1)
                mask = masks[:, :, i]

                result = {
                    "image_id": image_id,
                    "category_id": dataset.get_source_class_id(class_id, "coco"),
                    "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                    "score": score,
                    "segmentation": maskUtils.encode(np.asfortranarray(mask))
                }
                results.append(result)
        return results