from typing import Any, Optional
from pathlib import Path
from dataclasses import dataclass, InitVar
import msgspec

class _Image(msgspec.Struct):
    id:int
    file_name:str
    path:str|None=None
    annotations:'list[_Annotation]'=[]
    categories:'list[_Category]'=[]


class _Annotation(msgspec.Struct):
    id:int
    image_id:int
    category_id:int
    segmentation:list[Any]
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


class _COCO(msgspec.Struct):
    images: list[ _Image ]
    annotations:list[ _Annotation ]
    categories:list[ _Category ]


@dataclass
class Dataset:
    coco:InitVar[ _COCO ]

    def __post_init__(self, coco: _COCO):
        self.__coco = coco
        self.__cat_name_id_dict:dict[str,int] = {}
        self.__id_cat_name_dict:dict[int,str] = {}

        for i,cat in enumerate( self.categories ):
            self.__cat_name_id_dict[cat.name] = i
            self.__id_cat_name_dict[i] = cat.name

    @staticmethod
    def from_json(json_path:str, image_dir:str, include_classes:list[int]|None=None, exclude_classes:list[int]|None=None):
        if ( bool( include_classes ) & bool( exclude_classes ) ):
            raise ValueError(f'You must enter either \"include_classes\" or \"exclude_classes\"')
        
        with open(json_path, 'r') as f:
            coco = msgspec.json.decode(f.read(), type=_COCO)
        
        include_classes = Dataset.__get_include_classes(coco, include_classes)
        exclude_classes = Dataset.__get_exclude_classes(coco, exclude_classes)
        
        img_dict:dict[int,_Image] = {}
        for img in coco.images:
            img_id = img.id
            del img.id
            img_dict[img_id] = img
            img.path = (Path(image_dir) / img.file_name).as_posix()

        cat_dict:dict[int,_Category] = {}
        deleted_cat_id = []
        for cat in coco.categories:
            if ( cat.name in include_classes ) & (cat.name not in exclude_classes):
                cat_id = cat.id
                del cat.id
                cat_dict[cat_id] = cat
            else:
                deleted_cat_id.append(cat.id)
        coco.categories = list( cat_dict.values() )
        

        ann_dict:dict[int,_Annotation] = {}
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

                ann_id = ann.id
                del ann.id, ann.category_id, ann.image_id
                ann_dict[ann_id] = ann
            except KeyError:
                if ann.category_id in deleted_cat_id:
                    continue
                print(f'There was a problem loading annotation_id:{ann.id}. image_id:{ann.image_id} or category_id:{ann.category_id} does not exist. Proceed without loading.')
        
        return Dataset(coco)


    def __add__(self, one:'Dataset'):
        images = self.__coco.images + one.images
        categories = self.__coco.categories + one.categories
        annotations = self.__coco.annotations + one.annotations
        coco = _COCO(images=images, categories=categories,annotations=annotations)
        return Dataset(coco)
    

    @staticmethod
    def __get_include_classes(coco: _COCO, include_classes:list[str]|None) -> list[str]:
        json_classes = { cat.name for cat in coco.categories }
        if include_classes is None:
            return list(json_classes)

        include_classes = set(include_classes)
        if remain:=(include_classes - json_classes):
            raise ValueError(f'include classes, {remain}, are not found in dataset')
        
        return list(include_classes)


    @staticmethod
    def __get_exclude_classes(coco:_COCO, exclude_classes:list[str]|None) -> list[str]:
        if exclude_classes is None:
            return []

        json_classes = { cat.name for cat in coco.categories }
        exclude_classes = set(exclude_classes)
        if remain:=(exclude_classes - json_classes):
            raise ValueError(f'exclude classes, {remain}, are not found in dataset')
        return exclude_classes
    

    def get_loader_class_id(self, cat_name:str):
        return self.__cat_name_id_dict[cat_name]

    def get_cat_name(self, loader_class_id:int):
        return self.__id_cat_name_dict[loader_class_id]
    

    @property
    def images(self):
        return self.__coco.images

    @property
    def categories(self):
        return self.__coco.categories

    @property
    def annotations(self):
        return self.__coco.annotations
