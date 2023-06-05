import unittest
from MRCNN.data.dataset import _msgspecCOCO, Dataset

class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.json_path = 'test/source/instances_train_fix.json'

    def test_parse_coco_json(self):
        from msgspec.json import decode

        with open(self.json_path, 'r') as f:
            coco:_msgspecCOCO = decode(f.read(),type=_msgspecCOCO)
        
        self.assertTrue(bool(coco.annotations))
        self.assertTrue(bool(coco.images))
        self.assertTrue(bool(coco.categories))
    

    def test_load_dataset_1(self):
        dataset = Dataset.from_json(self.json_path,'./source/images')

        self.assertIsNotNone(dataset.annotations)
        self.assertIsNotNone(dataset.images)
        self.assertIsNotNone(dataset.categories)

    def test_load_dataset_2(self):
        self.assertRaises(ValueError, lambda : Dataset.from_json(self.json_path,'./source/images', include_classes=[1,2,3], exclude_classes=[4,5]))

    def test_load_dataset_3(self):
        include_classes = ["crack", "leakage", "peeling"]
        dataset = Dataset.from_json(self.json_path,'./source/images', include_classes=include_classes)
        self.assertEqual(set(include_classes),{cat.name for cat in dataset.categories})

    def test_load_dataset_4(self):
        exclude_classes = ["crack", "leakage", "peeling"]
        dataset = Dataset.from_json(self.json_path,'./source/images', exclude_classes=exclude_classes)
        self.assertFalse(set(exclude_classes).intersection({cat.name for cat in dataset.categories}))
    
    def test_add_dataset(self):
        dataset1 = Dataset.from_json(self.json_path,'./source/images')
        dataset2 = Dataset.from_json('test/source/test.json','./source/images')

        dataset3 = dataset1 + dataset2
        self.assertEqual(len(dataset3.annotations), len(dataset1.annotations)+len(dataset2.annotations))
        self.assertEqual(len(dataset3.images), len(dataset1.images)+len(dataset2.images))
        self.assertEqual(len(dataset3.categories), len(dataset1.categories)+len(dataset2.categories))
    
    def test_exist_coco(self):
        dataset = Dataset.from_json(self.json_path,'./source/images')

        self.assertTrue(dataset.coco.anns)
        self.assertTrue(dataset.coco.cats)
        self.assertTrue(dataset.coco.imgs)


