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
    images: list[ _Image ]|dict[int,_Image]
    annotations:list[ _Annotation ]|dict[int,_Annotation]
    categories:list[ _Category ]|dict[int,_Category]


@dataclass
class Dataset:
    json_path:InitVar[ str ]
    image_dir:InitVar[ str ]
    include_classes:InitVar[list[int]|None]=None
    exclude_classes:InitVar[list[int]|None]=None


    def __post_init__(self, json_path:str, image_dir:str, include_classes:list[int]|None, exclude_classes:list[int]|None):
        if ( bool( include_classes ) & bool( exclude_classes ) ):
            raise ValueError(f'You must enter either \"include_classes\" or \"exclude_classes\"')
        
        with open(json_path, 'r') as f:
            self.__coco = msgspec.json.decode(f.read(), type=_COCO)
        
        include_classes = self.__get_include_classes(include_classes)
        exclude_classes = self.__get_exclude_classes(exclude_classes)
        
        new_img = {}
        for img in self.__coco.images:
            img_id = img.id
            del img.id
            new_img[img_id] = img
            img.path = (Path(image_dir) / img.file_name).as_posix()
        self.__coco.images = new_img

        new_cat = {}
        for cat in self.__coco.categories:
            if ( cat.id in include_classes ) & (cat.id not in exclude_classes):
                cat_id = cat.id
                del cat.id
                new_cat[cat_id] = cat
        self.__coco.categories = new_cat

        new_ann = {}
        for ann in self.__coco.annotations:
            try:
                img = self.__coco.images[ann.image_id]
                cat = self.__coco.categories[ann.category_id]
                ann.image = img
                ann.category = cat

                img.annotations.append(ann)
                img.categories.append(cat)
                cat.annotations.append(ann)
                cat.images.append(img)

                ann_id = ann.id
                del ann.id
                new_ann[ann_id] = ann
            except KeyError:
                print(f'There was a problem loading annotation_id:{ann.id}. image_id:{ann.image_id} or category_id:{ann.category_id} does not exist. Proceed without loading.')

        self.__coco.annotations = new_ann
    

    def __get_include_classes(self, include_classes:list[int]|None) -> list[int]:
        json_classes = { cat.id for cat in self.__coco.categories }
        if include_classes is None:
            return list(json_classes)

        include_classes = set(include_classes)
        if remain:=(include_classes - json_classes):
            raise ValueError(f'include classes, {remain}, are not found in dataset')
        
        return list(include_classes)


    def __get_exclude_classes(self, exclude_classes:list[int]|None) -> list[int]:
        if exclude_classes is None:
            return []

        json_classes = { cat.id for cat in self.__coco.categories }
        exclude_classes = set(exclude_classes)
        if remain:=(exclude_classes - json_classes):
            raise ValueError(f'exclude classes, {remain}, are not found in dataset')
        return exclude_classes
    

    @property
    def images(self):
        return self.__coco.images

    @property
    def categories(self):
        return self.__coco.categories

    @property
    def annotations(self):
        return self.__coco.annotations
