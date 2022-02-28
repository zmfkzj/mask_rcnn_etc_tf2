from pathlib import Path
import tensorflow.keras as keras
import numpy as np
import pandas as pd

from MRCNN.detector import Detector
from MRCNN.config import Config
from MRCNN.data.data_loader import CocoDataset
from MRCNN.data.data_generator import data_generator

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from collections import OrderedDict

class Evaluator(Detector):
    def __init__(self, model, gt_image_dir, gt_json_path, config: Config = Config(), conf_thresh=0.25, iou_thresh=0.5) -> None:
        self.coco = COCO(gt_json_path)
        self.config = config
        self.gt_image_dir = gt_image_dir
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.classes = OrderedDict({cat_info['id']:cat_info['name'] for cat_info in self.coco.cats.values()})
        super().__init__(model, dict(self.classes), config)


    
    def eval(self, save_dir):
        detections =  self.detect(self.gt_image_dir, shuffle=True, limit_step=self.config.VALIDATION_STEPS)

        results_per_class = [{} {} {}]
        for cat_id, cat_name in self.classes.items():
            true, pred = self.get_state(cat_name, detections)
            metrics = [keras.metrics.AUC(curve='PR'), 
                        keras.metrics.Recall(thresholds=self.conf_thresh), 
                        keras.metrics.Precision(thresholds=self.conf_thresh)]
            for j, metric in enumerate(metrics):
                metric.update_state(true, pred)
                results_per_class[j][cat_name] = np.round(metric.result().numpy(), 4)

        for metric in results_per_class:
            for cat_name, metics in results_per_class.items():
                results_per_class[cat_id]=np.array(metrics)
        results_per_class[3] = self.cal_F1(results_per_class)
    
        results_for_all = [np.mean(r) for r in results_per_class]

        df_per_class = pd.DataFrame(results_per_class, index=self.classes, columns=[f'mAP{self.iou_thresh}',f'Recall{self.conf_thresh}',f'Precision{self.conf_thresh}',f'F1-Score{self.conf_thresh}'])
        df_for_all = pd.DataFrame(results_for_all, columns=[f'mAP{self.iou_thresh}',f'Recall{self.conf_thresh}',f'Precision{self.conf_thresh}',f'F1-Score{self.conf_thresh}'])
        with pd.ExcelWriter(Path(save_dir)/'results.xlsx') as writer:
            df_per_class.to_excel(writer, sheet_name='per_class')
            df_for_all.to_excel(writer, sheet_name='for_all')

        return self.classes, results_per_class, results_for_all

    
    def get_state(self, cat, detections):
        '''
        detections's keys: "path"(related path), "rois"(x1,y1,x2,y2), "classes", "scores", "masks"
        '''
        specific_detections = []
    
    def cal_map(self):

        pass

    def cal_racall(self):
        pass

    def cal_precision(self):
        pass

    def cal_F1(self, results):
        '''
        results = [map, recall, precision]
        '''
        f1 = 2*(results[2]*results[1])/(results[2]+results[1])
        return round(f1,4) if not np.isnan(f1) else np.nan

    def build_coco_results(self, detections):
        """Arrange resutls to match COCO specs in http://cocodataset.org/#format
        """
        pathes, rois, classes, scores, masks = detections

        dataset, image_ids, rois, class_ids, scores, masks
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


    def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
        """Runs official COCO evaluation.
        dataset: A Dataset object with valiadtion data
        eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
        limit: if not 0, it's the number of images to use for evaluation
        """
        # Pick COCO images from the dataset
        image_ids = image_ids or dataset.image_ids

        # Limit to a subset
        if limit:
            image_ids = image_ids[:limit]

        # Get corresponding COCO image IDs.
        coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

        t_prediction = 0
        t_start = time.time()

        results = []
        for i, image_id in enumerate(image_ids):
            # Load image
            image = dataset.load_image(image_id)

            # Run detection
            t = time.time()
            r = model.detect([image], verbose=0)[0]
            t_prediction += (time.time() - t)

            # Convert results to COCO format
            # Cast masks to uint8 because COCO tools errors out on bool
            image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                            r["rois"], r["class_ids"],
                                            r["scores"],
                                            r["masks"].astype(np.uint8))
            results.extend(image_results)

        # Load results. This modifies results with additional attributes.
        coco_results = coco.loadRes(results)

        # Evaluate
        cocoEval = COCOeval(coco, coco_results, eval_type)
        cocoEval.params.imgIds = coco_image_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        print("Prediction time: {}. Average {}/image".format(
            t_prediction, t_prediction / len(image_ids)))
        print("Total time: ", time.time() - t_start)
        
