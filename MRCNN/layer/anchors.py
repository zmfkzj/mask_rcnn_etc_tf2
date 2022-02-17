import tensorflow as tf
import tensorflow.keras.layers as KL

class AnchorsLayer(KL.Layer):

    def __init__(self, anchors, **kwargs):
        super(AnchorsLayer, self).__init__(**kwargs)
        self.anchors = anchors
    
    def call(self, image_placeholder):
        return self.anchors