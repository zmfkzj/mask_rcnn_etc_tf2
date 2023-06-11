import tensorflow as tf
import keras.api._v2.keras as K
import keras.api._v2.keras.layers as KL
import numpy as np
import tensorflow_addons as tfa

from MRCNN.model_utils.miscellenous_graph import BatchPackGraph

class SmoothL1Loss(KL.Layer):
    @tf.function
    def call(self, y_true, y_pred):
        """Implements Smooth-L1 loss.
        y_true and y_pred are typically: [N, 4], but could be any shape.
        """

        diff = tf.cast(tf.abs(y_true - y_pred), tf.float32)
        less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)
        loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
        loss = tf.cast(loss, tf.float16)
        return loss


class RpnClassLossGraph(KL.Layer):
    # def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
    #     super().__init__(trainable, name, dtype, dynamic, **kwargs)
    #     self.focal_loss = tf.losses.BinaryFocalCrossentropy(apply_class_balancing=True, from_logits=True, reduction=tf.losses.Reduction.NONE)


    @tf.function
    def call(self, rpn_match, rpn_class_logits):
        """RPN anchor classifier loss.

        rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
                -1=negative, 0=neutral anchor.
        rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for BG/FG.
        """
        # Squeeze last dim to simplify
        # rpn_match = tf.squeeze(rpn_match, -1)
        # Get anchor classes. Convert the -1/+1 match to 0/1 values.
        anchor_class = tf.cast(tf.equal(rpn_match, 1), tf.int32)
        # Positive and Negative anchors contribute to the loss,
        # but neutral anchors (match value = 0) don't.
        indices = tf.where(tf.not_equal(rpn_match, 0))
        # Pick rows that contribute to the loss and filter out the rest.
        rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
        anchor_class = tf.gather_nd(anchor_class, indices)
        # Cross entropy loss
        loss = K.losses.sparse_categorical_crossentropy(anchor_class, rpn_class_logits, from_logits=True)
        # anchor_class = tf.one_hot(anchor_class,2)
        # loss = self.focal_loss(anchor_class, rpn_class_logits)
        loss = tf.cond(tf.size(loss) > 0, lambda: tf.reduce_mean(loss), lambda: tf.constant(0.0, dtype=tf.float16))
        return loss


class RpnBboxLossGraph(KL.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.batch_pack = BatchPackGraph()
        self.smooth_l1 = SmoothL1Loss()


    @tf.function
    def call(self, target_bbox, rpn_match, rpn_bbox, batch_size):
        """Return the RPN bounding box loss graph.

        config: the model config object.
        target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
            Uses 0 padding to fill in unsed bbox deltas.
        rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
                -1=negative, 0=neutral anchor.
        rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
        """
        # Positive anchors contribute to the loss, but negative and
        # neutral anchors (match value of 0 or -1) don't.
        # rpn_match = tf.squeeze(rpn_match, -1)
        indices = tf.where(tf.equal(rpn_match, 1))

        # Pick bbox deltas that contribute to the loss
        rpn_bbox = tf.gather_nd(rpn_bbox, indices)

        # Trim target bounding box deltas to the same length as rpn_bbox.
        batch_counts = tf.reduce_sum(tf.cast(tf.equal(rpn_match, 1), tf.int32), axis=1)
        target_bbox = self.batch_pack(target_bbox, batch_counts, batch_size)

        loss = self.smooth_l1(target_bbox, rpn_bbox)
        
        loss = tf.cond(tf.size(loss) > 0, lambda:tf.reduce_mean(loss), lambda:tf.constant(0.0, dtype=tf.float16))
        # loss = tf.cond(tf.size(loss) > 0, lambda:tf.reduce_mean(loss), lambda:np.nan)
        return loss


class MrcnnClassLossGraph(KL.Layer):
    # def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
    #     super().__init__(trainable, name, dtype, dynamic, **kwargs)
    #     self.focal_loss = tf.losses.BinaryFocalCrossentropy(apply_class_balancing=True, from_logits=True, reduction=tf.losses.Reduction.NONE)


    @tf.function
    def call(self, target_class_ids, pred_class_logits):
        """Loss for the classifier head of Mask RCNN.

        target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
            padding to fill in the array.
        pred_class_logits: [batch, num_rois, num_classes]
        """
        # During model building, Keras calls this function with
        # target_class_ids of type float32. Unclear why. Cast it
        # to int to get around it.
        target_class_ids = tf.cast(target_class_ids, tf.int32)

        # Loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_class_ids, logits=pred_class_logits)

        # Computer loss mean. Use only predictions that contribute
        # to the loss to get a correct mean.
        loss = tf.reduce_mean(loss)
        return loss


class MrcnnBboxLossGraph(KL.Layer):
    @tf.function
    def call(self, target_bbox, target_class_ids, pred_bbox):
        """Loss for Mask R-CNN bounding box refinement.

        target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
        target_class_ids: [batch, num_rois]. Integer class IDs.
        pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
        """
        # Reshape to merge batch and roi dimensions for simplicity.
        target_class_ids = tf.reshape(target_class_ids, (-1,))
        target_bbox = tf.reshape(target_bbox, (-1, 4))
        pred_bbox = tf.reshape(pred_bbox, (-1, tf.shape(pred_bbox)[2], 4),)

        # Only positive ROIs contribute to the loss. And only
        # the right class_id of each ROI. Get their indices.
        positive_roi_ix = tf.cast(tf.where(target_class_ids > 0)[:, 0], tf.int32)
        positive_roi_class_ids = tf.cast(
            tf.gather(target_class_ids, positive_roi_ix), tf.int32)
        indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

        # Gather the deltas (predicted and true) that contribute to loss
        target_bbox = tf.gather(target_bbox, positive_roi_ix)
        pred_bbox = tf.gather_nd(pred_bbox, indices)

        # Smooth-L1 Loss
        loss = tf.cond(tf.size(target_bbox) > 0,
                       lambda: SmoothL1Loss()(y_true=target_bbox, y_pred=pred_bbox),
                       lambda: tf.constant(0.0, dtype=tf.float16))
        loss = tf.reduce_mean(loss)
        return loss


class MrcnnMaskLossGraph(KL.Layer):
    @tf.function
    def call(self, target_masks, target_class_ids, pred_masks):
        """Mask binary cross-entropy loss for the masks head.

        target_masks: [batch, num_rois, height, width].
            A float32 tensor of values 0 or 1. Uses zero padding to fill array.
        target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
        pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                    with values from 0 to 1.
        """
        # Reshape for simplicity. Merge first two dimensions into one.
        target_class_ids = tf.reshape(target_class_ids, (-1,))
        mask_shape = tf.shape(target_masks)
        target_masks = tf.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
        pred_shape = tf.shape(pred_masks)
        pred_masks = tf.reshape(pred_masks,
                            (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
        # Permute predicted masks to [N, num_classes, height, width]
        pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

        # Only positive ROIs contribute to the loss. And only
        # the class specific mask of each ROI.
        positive_ix = tf.cast(tf.where(target_class_ids > 0)[:, 0], tf.int32)
        positive_class_ids = tf.cast(
            tf.gather(target_class_ids, positive_ix), tf.int32)
        indices = tf.stack([positive_ix, positive_class_ids], axis=1)

        # Gather the masks (predicted and true) that contribute to loss
        y_true = tf.gather(target_masks, positive_ix)
        y_pred = tf.gather_nd(pred_masks, indices)

        # Compute binary cross entropy. If no positive ROIs, then return 0.
        # shape: [batch, roi, num_classes]
        loss = tf.cond(tf.size(y_true) > 0,
                       lambda: K.losses.binary_crossentropy(y_true, y_pred),
                       lambda: tf.constant(0.0, dtype=tf.float16))
        loss = tf.reduce_mean(loss)
        return loss