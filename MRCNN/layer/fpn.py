import tensorflow as tf
import tensorflow_models as tfm
import keras.api._v2.keras.layers as KL
import keras.api._v2.keras as keras

from MRCNN.config import Config


class FPN(KL.Layer):
    def __init__(self, config:Config, backbone_output, nas_train=False):
        super().__init__(name='fpn_model')
        self.nas_train = nas_train
        self.config = config

        output_specs = {str(i+2):t.get_shape() for i,t in enumerate(backbone_output.values())}
        if self.config.FPN == 'FPN':
            self.fpn = tfm.vision.decoders.FPN(input_specs=output_specs, 
                                                num_filters=config.TOP_DOWN_PYRAMID_SIZE,
                                                min_level=self.config.FPN_MIN_LEVEL,
                                                max_level=self.config.FPN_MAX_LEVEL)
        elif self.config.FPN == 'NASFPN':
            self.fpn = tfm.vision.decoders.NASFPN(input_specs=output_specs,
                                                num_filters=config.TOP_DOWN_PYRAMID_SIZE,
                                                min_level=self.config.FPN_MIN_LEVEL,
                                                max_level=self.config.FPN_MAX_LEVEL)
        else:
            raise ValueError('Invalid FPN')
        
    
    @tf.function
    def call(self, data):
        if self.nas_train and self.config.FPN=='NASFPN':
            return self.fpn.call(data)
        else:
            return self.fpn(data)
    

    def set_nas_train_mode(self, mode: bool):
        self.nas_train = mode
