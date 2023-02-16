from dataclasses import InitVar
from operator import itemgetter
from typing import Any, Optional, Union
from pycocotools.coco import COCO
from pydantic.dataclasses import dataclass
from pydantic import BaseModel
import json
import pathlib


class Segmentation(BaseModel):
    counts:list[int]
    size:list[int]


class Category(BaseModel):
    id:int
    name:str
    supercategory:str


class Image(BaseModel):
    id:int
    width:int
    height:int
    file_name:str
    path:Optional[str] = None


class Annotation(BaseModel):
    id:int
    image_id:int
    category_id:int
    segmentation:Union[list[list[float]],Segmentation]
    bbox:list[float]
    iscrowd:int
    score:float
    area:Optional[float] = None


class CocoJson(BaseModel):
    categories:list[Category]
    images:list[Image]
    annotations:list[Annotation]


@dataclass
class Dataset:
    json_path:str
    image_path:str

    def __post_init__(self):
        self.coco = COCO(self.json_path)

        with open(self.json_path, 'r') as f:
            anno = json.load(f)
        self.anno = CocoJson(**anno)
        self.anno.categories.sort(key=lambda item: item.id)
        for img in self.anno.images:
            img.path = (pathlib.Path(self.image_path) / img.file_name).as_posix()
    

    def get_source_class_id(self, dataloader_class_id):
        return self.anno.categories[dataloader_class_id-1] 
    

    def get_dataloader_class_id(self, source_class_id):
        for i, cat in enumerate(self.anno.categories):
            if cat.id==source_class_id:
                return i+1
        raise ValueError("dataset 내에 source_class_id가 존재하지 않습니다.")

    def __hash__(self) -> int:
        return hash((self.json_path, self.image_path))



