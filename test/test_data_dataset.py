import unittest
from MRCNN.data.dataset import Dataset


class TestCocoDataset(unittest.TestCase):
    def test_make_dataset(self):
        dataset = Dataset('instances_test.json', '~/host/dataset/5_coco_merge/images')
        self.assertTrue(dataset.anno.annotations)
        self.assertTrue(dataset.anno.categories)
        self.assertTrue(dataset.anno.images)
