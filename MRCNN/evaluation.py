import numpy as np
import time

from MRCNN.data_loader import CocoDataset

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

class Evaluator:
    def __init__(self, model, dataset, tensorboard=None) -> None:
        self.model = model
        self.dataset = dataset

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
                    "category_id": self.dataset.get_source_class_id(class_id, "coco"),
                    "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                    "score": score,
                    "segmentation": maskUtils.encode(np.asfortranarray(mask))
                }
                results.append(result)
        return results


    def evaluate(self, gt_json, limit=0, image_ids=None):
        """Runs official COCO evaluation.
        dataset: A Dataset object with valiadtion data
        eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
        limit: if not 0, it's the number of images to use for evaluation
        """
        # Pick COCO images from the dataset
        image_ids = image_ids or self.dataset.image_ids

        # Limit to a subset
        if limit:
            image_ids = image_ids[:limit]

        # Get corresponding COCO image IDs.
        coco_image_ids = [self.dataset.image_info[id]["id"] for id in image_ids]

        t_prediction = 0
        t_start = time.time()

        results = []
        for i, image_id in enumerate(image_ids):
            # Load image
            image = self.dataset.load_image(image_id)

            # Run detection
            t = time.time()
            r = self.model.detect([image], verbose=0)[0]
            t_prediction += (time.time() - t)

            # Convert results to COCO format
            # Cast masks to uint8 because COCO tools errors out on bool
            image_results = self.build_coco_results(coco_image_ids[i:i + 1],
                                                    r["rois"], r["class_ids"],
                                                    r["scores"],
                                                    r["masks"].astype(np.uint8))
            results.extend(image_results)

        # Load results. This modifies results with additional attributes.
        coco = COCO(gt_json)
        coco_results = coco.loadRes(results)

        # Evaluate
        cocoEval = COCOeval(coco, coco_results, 'segm')
        cocoEval.params.imgIds = coco_image_ids
        ious = cocoEval.computeIoU()

        print("Prediction time: {}. Average {}/image".format(
            t_prediction, t_prediction / len(image_ids)))
        print("Total time: ", time.time() - t_start)