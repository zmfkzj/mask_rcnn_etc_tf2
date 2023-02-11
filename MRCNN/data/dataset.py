from dataclasses import InitVar
from operator import itemgetter
from typing import Optional, Union
from pycocotools.coco import COCO
from pydantic.dataclasses import dataclass
import json
import pathlib


@dataclass
class Segmentation:
    counts:list[int]
    size:list[int]


@dataclass
class Category:
    id:int
    name:str
    supercategory:str


@dataclass
class Image:
    id:int
    width:int
    height:int
    file_name:str
    path:Optional[str] = None


@dataclass
class Annotation:
    id:int
    image_id:int
    category_id:int
    segmentation:Union[list[list[float]],Segmentation]
    area:float
    bbox:list[float]
    iscrowd:int
    score:float


@dataclass
class CocoJson:
    categories:list[Category]
    images:list[Image]
    annotations:list[Annotation]


@dataclass
class Dataset:
    json_path:InitVar[str]
    image_path:InitVar[str]

    def __post_init__(self, json_path, image_path):
        self.coco = COCO(json_path)

        with open(json_path, 'r') as f:
            anno = json.load(f)
        self.anno = CocoJson(**anno)
        self.anno.categories.sort(key=lambda item: item.id)
        for img in self.anno.images:
            img.path = (pathlib.Path(image_path) / img.file_name).as_posix()
    

    def get_source_class_id(self, dataloader_class_id):
        return self.anno.categories[dataloader_class_id-1] 
    

    def get_dataloader_class_id(self, source_class_id):
        for i, cat in enumerate(self.anno.categories):
            if cat.id==source_class_id:
                return i+1
        raise ValueError("dataset 내에 source_class_id가 존재하지 않습니다.")




