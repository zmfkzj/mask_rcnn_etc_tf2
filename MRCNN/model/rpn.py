import tensorflow as tf
import keras.api._v2.keras.layers as KL
import keras.api._v2.keras.models as KM
import keras.api._v2.keras as keras


class RPN(KM.Model):
    def __init__(self, anchor_stride, anchors_per_location, *args, **kwdargs):
        super().__init__(*args, **kwdargs)

        # TODO: check if stride of 2 causes alignment issues if the feature map
        # is not even.
        # Shared convolutional base of the RPN
        self.rpn_conv_shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu', strides=anchor_stride, name='rpn_conv_shared')
        self.rpn_shared_bn = KL.BatchNormalization(name='rpn_shared_bn')

        # Anchor Score. [batch, height, width, anchors per location * 2].
        self.rpn_class_raw = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid', activation='linear', name='rpn_class_raw')

        # Bounding box refinement. [batch, H, W, anchors per location * depth]
        # where depth is [x, y, log(w), log(h)]
        self.rpn_bbox_pred = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid", activation='linear', name='rpn_bbox_pred')


    def call(self, feature_map):
        """Builds the computation graph of Region Proposal Network.

        feature_map: backbone features [batch, height, width, depth]
        anchors_per_location: number of anchors per pixel in the feature map
        anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                    every pixel in the feature map), or 2 (every other pixel).

        Returns:
            rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
            rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
            rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                    applied to anchors.
        """
        # TODO: check if stride of 2 causes alignment issues if the feature map
        # is not even.
        # Shared convolutional base of the RPN
        shared = self.rpn_conv_shared(feature_map)
        # shared = self.rpn_shared_bn(shared)
        # shared = KL.ReLU()(shared)

        # Anchor Score. [batch, height, width, anchors per location * 2].
        x = self.rpn_class_raw(shared)

        # Reshape to [batch, anchors, 2]
        rpn_class_logits = tf.reshape(x, [tf.shape(x)[0], -1, 2])

        # # Softmax on last dimension of BG/FG.
        rpn_probs = KL.Activation("softmax", name="rpn_class_xxx")(rpn_class_logits)

        # Bounding box refinement. [batch, H, W, anchors per location * depth]
        # where depth is [x, y, log(w), log(h)]
        x = self.rpn_bbox_pred(shared)

        # Reshape to [batch, anchors, 4]
        rpn_bbox = tf.reshape(x, [tf.shape(x)[0], -1, 4])

        return [rpn_class_logits, rpn_probs, rpn_bbox]