import keras.api._v2.keras.layers as KL
import tensorflow as tf



class Attentive(KL.Layer):
    def call(self, feature_map, attentions):
        """apply attentive vector

        Args:
            feature_map (_type_): [batch_size, None, None, TOP_DOWN_PYRAMID_SIZE]
            attentions (_type_): [NUM_CLASSES, TOP_DOWN_PYRAMID_SIZE]
        """

        feature_map = tf.expand_dims(feature_map, -2) #[batch_size, None, None, 1, TOP_DOWN_PYRAMID_SIZE]
        attentive_map = feature_map * attentions # [batch_size, None, None, NUM_CLASSES, TOP_DOWN_PYRAMID_SIZE]
        return attentive_map