import unittest
from MRCNN.data.dataset import _COCO, Dataset

class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.json_path = 'test/source/instances_train_fix.json'

    def test_parse_coco_json(self):
        from msgspec.json import decode

        with open(self.json_path, 'r') as f:
            coco:_COCO = decode(f.read(),type=_COCO)
        
        self.assertTrue(bool(coco.annotations))
        self.assertTrue(bool(coco.images))
        self.assertTrue(bool(coco.categories))
    

    def test_load_dataset_1(self):
        dataset = Dataset(self.json_path,'./source/images')

        self.assertIsNotNone(dataset.annotations)
        self.assertIsNotNone(dataset.images)
        self.assertIsNotNone(dataset.categories)
        self.assertEqual(type(dataset.annotations),dict)
        self.assertEqual(type(dataset.images),dict)
        self.assertEqual(type(dataset.categories),dict)
        self.assertRaises(AttributeError, lambda: dataset.annotations.popitem()[1].id)
        self.assertRaises(AttributeError, lambda: dataset.images.popitem()[1].id)
        self.assertRaises(AttributeError, lambda: dataset.categories.popitem()[1].id)

    def test_load_dataset_2(self):
        self.assertRaises(ValueError, lambda : Dataset(self.json_path,'./source/images', include_classes=[1,2,3], exclude_classes=[4,5]))

    def test_load_dataset_3(self):
        dataset = Dataset(self.json_path,'./source/images', include_classes=[1,2,3])
        self.assertEqual({1,2,3},{id for id in dataset.categories})

    def test_load_dataset_4(self):
        dataset = Dataset(self.json_path,'./source/images', exclude_classes=[1,2,3])
        self.assertFalse({1,2,3}.intersection({id for id in dataset.categories}))
