from typing import Any, Optional, Union
from pycocotools.coco import COCO
from pydantic.dataclasses import dataclass
import pathlib


@dataclass
class Dataset:
    json_path:str
    image_path:str

    def __post_init__(self):
        self.coco = COCO(self.json_path)

        for img in self.coco.dataset['images']:
            img['path'] = (pathlib.Path(self.image_path) / img['file_name']).as_posix()
        
        self.count_each_class_objects()
    

    def get_source_class_id(self, dataloader_class_id):
        return self.coco.dataset['categories'][dataloader_class_id-1]['id']
    

    def get_dataloader_class_id(self, source_class_id):
        for i, cat in enumerate(self.coco.dataset['categories']):
            if cat['id']==source_class_id:
                return i+1
        raise ValueError("dataset 내에 source_class_id가 존재하지 않습니다.")

    def __hash__(self) -> int:
        return hash((self.json_path, self.image_path))
    
    def __len__(self):
        return len(self.coco.dataset['images'])

    def count_each_class_objects(self):
        self.class_count:dict[int,int] = {}
        for dataset_class_id in self.coco.cats:
            anns = self.coco.getAnnIds(catIds=dataset_class_id)
            self.class_count[dataset_class_id] = len(anns)
        self.min_class_count:int = min(self.class_count.values())



