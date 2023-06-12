import tensorflow as tf
import tensorflow_models as tfm
import keras.api._v2.keras.layers as KL
import keras.api._v2.keras as keras

from MRCNN.config import Config


class FPN(KL.Layer):
    def __init__(self, config:Config, backbone_output):
        super().__init__(name='fpn_model')

        output_specs = {str(i+2):t.get_shape() for i,t in enumerate(backbone_output.values())}
        if config.FPN == 'FPN':
            self.fpn = tfm.vision.decoders.FPN(input_specs=output_specs, 
                                                num_filters=config.TOP_DOWN_PYRAMID_SIZE,
                                                min_level=3,
                                                max_level=7)
        elif config.FPN == 'NASFPN':
            self.fpn = tfm.vision.decoders.NASFPN(input_specs=output_specs,
                                                num_filters=config.TOP_DOWN_PYRAMID_SIZE,
                                                min_level=3,
                                                max_level=7)
        else:
            raise ValueError('Invalid FPN')
    
    def call(self, data):
        return self.fpn(data)