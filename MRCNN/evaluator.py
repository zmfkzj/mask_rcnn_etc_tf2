from pathlib import Path
import tensorflow.keras as keras
import numpy as np
import pandas as pd

from MRCNN.detector import Detector
from MRCNN.config import Config
from MRCNN.data.data_loader import CocoDataset

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from collections import OrderedDict,defaultdict

class Evaluator(Detector):
    def __init__(self, model, gt_image_dir, gt_json_path, config: Config = Config(), conf_thresh=0.25, iou_thresh=0.5) -> None:
        assert iou_thresh in np.arange(0.5,1,0.05)
        self.iou_idx = {iou:idx for iou,idx in zip(np.arange(0.5,1,0.05), range(10))}[iou_thresh]
        self.config = config
        self.gt_image_dir = gt_image_dir
        self.conf_thresh = conf_thresh
        self.dataset = CocoDataset()
        self.coco = self.dataset.load_coco(gt_image_dir, gt_json_path, return_coco=True)
        self.classes = [info['name'] for info in self.dataset.class_info]
        self.image_filename_id = {img['file_name']:img['id'] for img in self.coco.imgs.values()}
        super().__init__(model, self.classes, config)
        # self.eval(limit_step=3, iouType='bbox')

    def eval(self, save_dir=None, limit_step=-1, iouType='segm')->dict:
        detections =  self.detect(self.gt_image_dir, shuffle=True, limit_step=limit_step)

        results_per_class = defaultdict(OrderedDict)
        for class_id, cat_name in enumerate(self.classes):
            if cat_name=='BG':
                continue
            true, pred, sample_weight,mAP50 = self.get_state(self.dataset.get_source_class_id(class_id, 'coco'), detections, iouType=iouType)
            metrics = { 'recall':keras.metrics.Recall(thresholds=self.conf_thresh), 
                        'precision':keras.metrics.Precision(thresholds=self.conf_thresh)}
            for metric_name, metric_fn in metrics.items():
                metric_fn.reset_state()
                metric_fn.update_state(true, pred, sample_weight)
                results_per_class[metric_name][cat_name] = np.round(metric_fn.result().numpy(), 4)
            results_per_class['mAP'][cat_name] = mAP50


        results_per_class['F1-Score'] = self.cal_F1(results_per_class)
    
        results_for_all = {metric_name:np.mean(list(metric_per_class.values())) for metric_name, metric_per_class in results_per_class.items()}

        metrics_head = [f'mAP50',f'Recall{int(self.conf_thresh*100)}',f'Precision{int(self.conf_thresh*100)}',f'F1-Score{int(self.conf_thresh*100)}']
        df_per_class = pd.DataFrame(results_per_class).rename(columns=dict(zip(['mAP','recall','precision','F1-Score'],metrics_head)))
        df_for_all = pd.DataFrame({'total':results_for_all}).T.rename(columns=dict(zip(['mAP','recall','precision','F1-Score'],metrics_head)))

        if save_dir is not None:
            with pd.ExcelWriter(Path(save_dir)/'results.xlsx') as writer:
                df_per_class.to_excel(writer, sheet_name='per_class',encoding='euc-kr')
                df_for_all.to_excel(writer, sheet_name='for_all',encoding='euc-kr')

        return results_for_all

    
    def get_state(self, class_id, detections,iouType='segm'):
        '''
        detections's keys: "path"(related path), "rois"(x1,y1,x2,y2), "classes", "class_ids", "scores", "masks"
        '''
        assert iouType in ['bbox', 'segm']
        coco_detections = self.build_coco_results(detections)
        if not coco_detections:
            return [],[],[],0
        coco_results = self.coco.loadRes(coco_detections)
        coco_image_ids = [self.image_filename_id[det['path']] for det in detections]

        # Evaluate
        cocoEval = COCOeval(self.coco, coco_results,iouType=iouType)
        cocoEval.params.imgIds = coco_image_ids
        cocoEval.params.catIds = class_id
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        mAP50 = cocoEval.stats[1]
        true = []
        pred = []
        sample_weight = []
        for img in cocoEval.evalImgs:
            if img is not None:
                gtIds:list = img['gtIds']
                dtScores = img['dtScores']
                dtMatches = img['dtMatches'][self.iou_idx]

                _true, _pred, _sample_weight = zip(*([(1,1,0) if gtId in dtMatches else (1,0,1) for gtId in gtIds] 
                                                     + [(0,score,1) if gtId==0 else (1,score,1) for gtId, score in zip(dtMatches,dtScores)]))
                true.extend(_true)
                pred.extend(_pred)
                sample_weight.extend(_sample_weight)

        return true, pred, sample_weight,mAP50

    def cal_F1(self, results):
        '''
        results = {'map':{'cls1':val1, 'cls2':val2,...}, 
                    'recall':{'cls1':val1, 'cls2':val2,...}, 
                    'precision', {'cls1':val1, 'cls2':val2,...}}
        '''
        classes = results['precision'].keys()
        precision = np.array(list(results['precision'].values()))
        recall = np.array(list(results['recall'].values()))
        f1 = 2*(precision*recall)/(precision+recall)
        f1 = np.where(np.isnan(f1), np.nan, np.round(f1))
        return {cat_name:cat_f1 for cat_name, cat_f1 in zip(classes, f1)}

    def build_coco_results(self, detections):
        """Arrange resutls to match COCO specs in http://cocodataset.org/#format
        """

        results = []
        for det in detections:
            image_path, rois, _, class_ids, scores, masks = det.values()
            image_id = self.image_filename_id[image_path]
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