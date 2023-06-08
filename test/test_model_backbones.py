import unittest
from MRCNN.model.backbones import *

class TestModelBackbones(unittest.TestCase):
    def test_build_keras_backbone_1(self):
        model = backbones['EfficientNetV2B0']
        backbone = build_keras_backbone('EfficientNetV2B0')

        input_data = tf.zeros([2,1024,1024,3])
        outputs = backbone(input_data)
        self.assertEqual(4, len(outputs))

    def test_build_keras_backbone_2(self):
        backbone = build_keras_backbone('ResNet152V2')

        input_data = tf.zeros([2,1024,1024,3])
        outputs = backbone(input_data)
        self.assertEqual(4, len(outputs))

    def test_build_keras_backbone_3(self):
        backbone = build_keras_backbone('EfficientNetV2L')

        input_data = tf.zeros([2,1024,1024,3])
        outputs = backbone(input_data)
        self.assertEqual(4, len(outputs))
        
    def test_build_keras_backbone_4(self):
        backbone = build_keras_backbone('ResNet101')

        input_data = tf.zeros([2,1024,1024,3])
        outputs = backbone(input_data)
        self.assertEqual(4, len(outputs))
        
    def test_build_convnext_backbone_1(self):
        backbone = build_convnext_backbone('convnext_base')

        input_data = tf.zeros([2,1024,1024,3])
        outputs = backbone(input_data)
        self.assertEqual(4, len(outputs))