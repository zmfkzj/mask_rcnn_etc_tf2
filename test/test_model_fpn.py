import unittest
import tensorflow as tf

from MRCNN.config import Config
from MRCNN.model import FPN
from MRCNN.model.backbones import backbones
from MRCNN.utils import compute_backbone_shapes


class TestModelFpn(unittest.TestCase):
    def test_create_fpn(self):
        config = Config(FPN='FPN')
        input_size = 1024
        model_name = config.BACKBONE
        _, builder = backbones[model_name]

        backbone = builder(model_name)
        fpn = FPN(config, backbone.output)

        input_data = tf.zeros([2,input_size,input_size,3], tf.float16)

        endpoints = backbone(input_data)
        feats = fpn(endpoints)


        for level in range(3, 7 + 1):
          self.assertIn(str(level), feats)
          self.assertEqual(
              [2, input_size // 2**level, input_size // 2**level, config.TOP_DOWN_PYRAMID_SIZE],
              feats[str(level)].shape.as_list())