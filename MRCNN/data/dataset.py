from copy import copy, deepcopy
from typing import Any, Optional
from pathlib import Path
from dataclasses import dataclass, InitVar
import msgspec
from pycocotools.coco import COCO
from pycocotools.cocoeval import  COCOeval
import numpy as np
from msgspec.structs import asdict

class _Image(msgspec.Struct):
    id:int
    file_name:str
    path:str|None=None
    annotations:'list[_Annotation]'=[]
    categories:'list[_Category]'=[]


class _Segmentation(msgspec.Struct):
    counts:list[int]
    size:list[int]


class _Annotation(msgspec.Struct):
    id:int
    image_id:int
    category_id:int
    segmentation:list[list[float]]|_Segmentation
    area:float
    bbox:list[float]
    iscrowd:int 
    category:"_Category|None"=None
    image:_Image|None=None


class _Category(msgspec.Struct):
    id:int
    name:str
    annotations:list[_Annotation]=[]
    images:list[_Image]=[]


class _msgspecCOCO(msgspec.Struct):
    images: list[ _Image ]
    annotations:list[ _Annotation ]
    categories:list[ _Category ]


class CustomCOCOeval(COCOeval):
    def get_map(self, iouThr=None):
        areaRng='all'
        maxDets=100 
        p = self.params

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        # dimension of precision: [TxRxKxAxM]
        s = self.eval['precision']
        # IoU
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:,:,:,aind,mind]
        if len(s[s>-1])==0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s>-1])
        return mean_s


@dataclass
class Dataset:
    msgspec_coco:InitVar[ _msgspecCOCO ]

    def __post_init__(self, msgspec_coco: _msgspecCOCO):
        self.__msgspec_coco = msgspec_coco

        self.__catId_loaderId_dict:dict[int,int] = {}
        self.__loaderId_catId_dict:dict[int,int] = {}
        self.__catId_catName_dict:dict[int,str] = {}
        for i,cat in enumerate( self.categories ):
            self.__catId_loaderId_dict[cat.id] = i
            self.__loaderId_catId_dict[i] = cat.id
            self.__catId_catName_dict[cat.id] = cat.name
        
        self.__make_cocotools_coco()


    @staticmethod
    def from_json(json_path:str, image_dir:str, include_classes:list[int]|None=None, exclude_classes:list[int]|None=None):
        if ( bool( include_classes ) & bool( exclude_classes ) ):
            raise ValueError(f'You must enter either \"include_classes\" or \"exclude_classes\"')
        
        with open(json_path, 'r') as f:
            coco = msgspec.json.decode(f.read(), type=_msgspecCOCO)
        
        include_classes = Dataset.__get_include_classes(coco, include_classes)
        exclude_classes = Dataset.__get_exclude_classes(coco, exclude_classes)
        
        img_dict:dict[int,_Image] = {}
        for img in coco.images:
            img_dict[img.id] = img
            img.path = (Path(image_dir) / img.file_name).as_posix()

        cat_dict:dict[int,_Category] = {}
        deleted_cat_id = []
        for cat in coco.categories:
            if ( cat.name in include_classes ) & (cat.name not in exclude_classes):
                cat_dict[cat.id] = cat
            else:
                deleted_cat_id.append(cat.id)
        coco.categories = list( cat_dict.values() )
        

        new_ann = []
        for ann in coco.annotations:
            try:
                img = img_dict[ann.image_id]
                cat = cat_dict[ann.category_id]
                ann.image = img
                ann.category = cat

                img.annotations.append(ann)
                img.categories.append(cat)
                cat.annotations.append(ann)
                cat.images.append(img)

                new_ann.append(ann)

            except KeyError:
                if ann.category_id in deleted_cat_id:
                    continue
                print(f'There was a problem loading annotation_id:{ann.id}. image_id:{ann.image_id} or category_id:{ann.category_id} does not exist. Proceed without loading.')
        coco.annotations = new_ann
            
        coco = Dataset.__reassign_id(coco)
        return Dataset(coco)


    def __add__(self, one:'Dataset'):
        images = self.__msgspec_coco.images + one.images
        categories = self.__msgspec_coco.categories + one.categories
        annotations = self.__msgspec_coco.annotations + one.annotations
        coco = _msgspecCOCO(images=images, categories=categories,annotations=annotations)
        coco = self.__reassign_id(coco)
        return Dataset(coco)
    

    @staticmethod
    def __get_include_classes(coco: _msgspecCOCO, include_classes:list[str]|None) -> list[str]:
        json_classes = { cat.name for cat in coco.categories }
        if include_classes is None:
            return list(json_classes)

        include_classes = set(include_classes)
        if remain:=(include_classes - json_classes):
            raise ValueError(f'include classes, {remain}, are not found in dataset')
        
        return list(include_classes)


    @staticmethod
    def __get_exclude_classes(coco:_msgspecCOCO, exclude_classes:list[str]|None) -> list[str]:
        if exclude_classes is None:
            return []

        json_classes = { cat.name for cat in coco.categories }
        exclude_classes = set(exclude_classes)
        if remain:=(exclude_classes - json_classes):
            raise ValueError(f'exclude classes, {remain}, are not found in dataset')
        return exclude_classes
    

    @staticmethod
    def __reassign_id(msgspec_coco:_msgspecCOCO):
        for i, img in enumerate( msgspec_coco.images ):
            img.id = i+1
        
        for i, cat in enumerate( msgspec_coco.categories ):
            cat.id = i+1
        
        for i, ann in enumerate( msgspec_coco.annotations ):
            ann.id = i+1
            ann.category_id = ann.category.id
            ann.image_id = ann.image.id
        return msgspec_coco


    def __make_cocotools_coco(self):
        msgspec_coco = deepcopy(self.__msgspec_coco)
        for img in msgspec_coco.images:
            img.annotations = []
            img.categories = []
        
        for cat in msgspec_coco.categories:
            cat.images = []
            cat.annotations = []
        
        for ann in msgspec_coco.annotations:
            ann.category = None
            ann.image = None

        coco_dict = msgspec.to_builtins(msgspec_coco)

        self.__cocotools_coco = COCO()
        self.__cocotools_coco.dataset = coco_dict
        self.__cocotools_coco.createIndex()
    

    def evaluate(self, coco_results, param_image_ids, iou_threshold=None, eval_type='bbox'):
        """
        Args:
            coco_results (dict): coco result dictionary
            param_image_ids (list[int]): image_ids param
            iou_threshold (float|None): Defaults to 0.5.
            eval_type (str): bbox or segm. Defaults to 'bbox'.
        """        
        coco_results = self.coco.loadRes(coco_results)

        coco_eval = CustomCOCOeval(self.coco, coco_results, eval_type)
        coco_eval.params.imgIds = list(param_image_ids)
        coco_eval.evaluate()
        coco_eval.accumulate()
        mAP = coco_eval.get_map(iou_threshold)
        eval_imgs = coco_eval.evalImgs
        return mAP, eval_imgs

    
    def get_loader_class_id(self, cat_name:str):
        return self.__catId_loaderId_dict[cat_name]


    def get_cat_id(self, loader_class_id:int):
        return self.__loaderId_catId_dict[loader_class_id]
    

    def get_cat_name(self, cat_id:int):
        return self.__catId_catName_dict[cat_id]

    @property
    def images(self):
        return self.__msgspec_coco.images

    @property
    def categories(self):
        return self.__msgspec_coco.categories

    @property
    def annotations(self):
        return self.__msgspec_coco.annotations

    @property
    def coco(self) -> COCO:
        return self.__cocotools_coco
