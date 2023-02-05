import unittest
from MRCNN.utils import compute_backbone_shapes
from MRCNN.config import Config

import numpy as np
import keras.api._v2.keras as keras

class TestComputeBackboneShapes(unittest.TestCase):
    def test_compute_backbone_shapes_ResNet101_1024(self):
        config = Config()
        config.BACKBONE = keras.applications.ResNet101
        config.IMAGE_SHAPE = np.array([1024, 1024,3])
        shapes = compute_backbone_shapes(config)
        assert(np.all(shapes == np.array([[512, 512], [256, 256], [128, 128], [ 64,  64], [ 32,  32]])))

    def test_compute_backbone_shapes_ResNet152_1024(self):
        config = Config()
        config.BACKBONE = keras.applications.ResNet152
        config.IMAGE_SHAPE = np.array([1024, 1024,3])
        shapes = compute_backbone_shapes(config)
        assert(np.all(shapes == np.array([[512, 512], [256, 256], [128, 128], [ 64,  64], [ 32,  32]])))

    def test_compute_backbone_shapes_EfficientNetB7_1024(self):
        config = Config()
        config.BACKBONE = keras.applications.EfficientNetB7
        config.IMAGE_SHAPE = np.array([1024, 1024,3])
        shapes = compute_backbone_shapes(config)
        assert(np.all(shapes == np.array([[512, 512], [256, 256], [128, 128], [ 64,  64], [ 32,  32]])))

    def test_compute_backbone_shapes_ResNetRS200_1024(self):
        config = Config()
        config.BACKBONE = keras.applications.ResNetRS200
        config.IMAGE_SHAPE = np.array([1024, 1024,3])
        shapes = compute_backbone_shapes(config)
        assert(np.all(shapes == np.array([[512, 512], [256, 256], [128, 128], [ 64,  64], [ 32,  32]])))

    def test_compute_backbone_shapes_ResNet101_1056(self):
        config = Config()
        config.BACKBONE = keras.applications.ResNet101
        config.IMAGE_SHAPE = np.array([1056, 1056,3])
        shapes = compute_backbone_shapes(config)
        assert(np.all(shapes == np.array([[528, 528], [264, 264], [132, 132], [ 66,  66], [ 33,  33]])))

    def test_compute_backbone_shapes_ResNet152_1056(self):
        config = Config()
        config.BACKBONE = keras.applications.ResNet152
        config.IMAGE_SHAPE = np.array([1056, 1056,3])
        shapes = compute_backbone_shapes(config)
        assert(np.all(shapes == np.array([[528, 528], [264, 264], [132, 132], [ 66,  66], [ 33,  33]])))

    def test_compute_backbone_shapes_EfficientNetB7_1056(self):
        config = Config()
        config.BACKBONE = keras.applications.EfficientNetB7
        config.IMAGE_SHAPE = np.array([1056, 1056,3])
        shapes = compute_backbone_shapes(config)
        assert(np.all(shapes == np.array([[528, 528], [264, 264], [132, 132], [ 66,  66], [ 33,  33]])))

    def test_compute_backbone_shapes_ResNetRS200_1056(self):
        config = Config()
        config.BACKBONE = keras.applications.ResNetRS200
        config.IMAGE_SHAPE = np.array([1056, 1056,3])
        shapes = compute_backbone_shapes(config)
        assert(np.all(shapes == np.array([[528, 528], [264, 264], [132, 132], [ 66,  66], [ 33,  33]])))
