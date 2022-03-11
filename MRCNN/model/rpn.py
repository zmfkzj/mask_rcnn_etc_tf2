import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM


class RPN(KM.Model):
    def __init__(self, anchor_stride, anchors_per_location):
        super().__init__()

        # TODO: check if stride of 2 causes alignment issues if the feature map
        # is not even.
        # Shared convolutional base of the RPN
        self.conv1 = KL.Conv2D(512, (3, 3), padding='same', activation='relu', strides=anchor_stride, name='rpn_conv_shared')

        # Anchor Score. [batch, height, width, anchors per location * 2].
        self.conv2 = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid', activation='linear', name='rpn_class_raw')

        # Bounding box refinement. [batch, H, W, anchors per location * depth]
        # where depth is [x, y, log(w), log(h)]
        self.conv3 = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid", activation='linear', name='rpn_bbox_pred')

        # self.rpn_class_logits = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))
        # self.rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))
        

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
        shared = self.conv1(feature_map)

        # Anchor Score. [batch, height, width, anchors per location * 2].
        x = self.conv2(shared)

        # Reshape to [batch, anchors, 2]
        # rpn_class_logits = self.rpn_class_logits(x)
        rpn_class_logits = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

        # Softmax on last dimension of BG/FG.
        rpn_probs = KL.Activation( "softmax", name="rpn_class_xxx")(rpn_class_logits)

        # Bounding box refinement. [batch, H, W, anchors per location * depth]
        # where depth is [x, y, log(w), log(h)]
        x = self.conv3(shared)

        # Reshape to [batch, anchors, 4]
        # rpn_bbox = self.rpn_bbox(x)
        rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

        return [rpn_class_logits, rpn_probs, rpn_bbox]