from typing import Any, Optional, Union
from pycocotools.coco import COCO
from pydantic.dataclasses import dataclass
import pathlib


@dataclass
class Dataset:
    json_path: str
    image_path: str

    def __post_init__(self):
        self.coco = COCO(self.json_path)

        for img in self.coco.dataset['images']:
            img['path'] = (pathlib.Path(self.image_path) / img['file_name']).as_posix()

        self.count_each_class_objects()
        self.set_dataloader_class_list([])

    def get_source_class_id(self, dataloader_class_id: int) -> int:
        """Return the corresponding source class ID for a given dataloader class ID."""
        try:
            return self.dataloader_class_list[dataloader_class_id - 1]
        except IndexError:
            raise ValueError(f"The dataloader_class_id {dataloader_class_id} does not exist in the dataset.")

    def get_dataloader_class_id(self, source_class_id: int) -> int:
        """Return the corresponding dataloader class ID for a given source class ID."""
        for i, cat_id in enumerate(self.dataloader_class_list):
            if cat_id == source_class_id:
                return i + 1
        raise ValueError(f"The source_class_id {source_class_id} does not exist in the dataset.")

    def __hash__(self) -> int:
        return hash((self.json_path, self.image_path))

    def __len__(self) -> int:
        return len(self.coco.dataset['images'])

    def count_each_class_objects(self):
        """Count the number of objects for each class in the dataset."""
        self.class_count = {}
        for dataset_class_id in self.coco.cats:
            anns = self.coco.getAnnIds(catIds=dataset_class_id)
            self.class_count[dataset_class_id] = len(anns)
        self.min_class_count = min(self.class_count.values())
        self.max_class_count = max(self.class_count.values())

    def set_dataloader_class_list(self, novel_class_ids: list[int]):
        """Set the dataloader class list excluding the specified novel class IDs."""
        self.dataloader_class_list = [cat_id for cat_id in self.coco.cats if cat_id not in novel_class_ids]
