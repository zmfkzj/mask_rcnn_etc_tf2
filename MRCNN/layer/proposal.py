import numpy as np
import tensorflow as tf
import keras.api._v2.keras.layers as KL
from MRCNN import utils
from MRCNN.config import Config

@tf.function
def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [batch, N, (y1, x1, y2, x2)] boxes to update
    deltas: [batch, N, (dy, dx, log(dh), log(dw))] refinements to apply
    """
    # Convert to y, x, h, w
    height = boxes[..., 2] - boxes[..., 0]
    width = boxes[..., 3] - boxes[..., 1]
    center_y = boxes[..., 0] + 0.5 * height
    center_x = boxes[..., 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[..., 0] * height
    center_x += deltas[..., 1] * width
    height *= tf.exp(deltas[..., 2])
    width *= tf.exp(deltas[..., 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result

@tf.function
def clip_boxes_graph(boxes, window):
    """
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] in the form y1, x1, y2, x2
    """
    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


class ProposalLayer(KL.Layer):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.

    Inputs:
        rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    def __init__(self, nms_threshold, config:Config=None, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.nms_threshold = nms_threshold

    def call(self, inputs):
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]
        # Box deltas [batch, num_rois, 4]
        deltas = inputs[1]
        deltas = deltas * tf.cast(tf.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4]),tf.float32)
        # Anchors
        anchors = inputs[2]
        proposal_count = inputs[3]

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
        ix = tf.math.top_k(scores, pre_nms_limit, sorted=True, name="top_anchors",).indices
        scores = tf.gather(scores, ix, batch_dims=1, axis=1)
        deltas = tf.gather(deltas, ix, batch_dims=1, axis=1)
        pre_nms_anchors = tf.gather(anchors, ix, batch_dims=1, axis=1)
        # pre_nms_anchors = tf.cast(tf.vectorized_map(lambda x: denorm_boxes_graph(x,list(self.config.IMAGE_SHAPE)[:2]),pre_nms_anchors), tf.float32)

        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, y2, x2)]
        boxes = tf.map_fn(lambda x:apply_box_deltas_graph(*x),
                          [pre_nms_anchors,deltas], 
                          fn_output_signature=tf.TensorSpec(shape=(self.config.PRE_NMS_LIMIT, 4), dtype=tf.float32))

        # Clip to image boundaries. Since we're in normalized coordinates,
        # window = tf.stack([0., 0., *list(self.config.IMAGE_SHAPE-1)[:2]])
        window = tf.stack([0., 0., 1., 1.])
        boxes = tf.vectorized_map(lambda boxes:clip_boxes_graph(boxes, window),boxes)

        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy
        # for small objects, so we're skipping it.

        # Non-max suppression
        def nms(inputs):
            boxes = inputs[0]
            scores = inputs[1]
            indices = tf.image.non_max_suppression(
                boxes, scores, proposal_count,
                self.nms_threshold, name="rpn_non_max_suppression")
            proposals = tf.gather(boxes, indices)
            # Pad if needed
            padding = tf.maximum(proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals
        proposals = tf.map_fn(nms,(boxes, scores),
                              fn_output_signature=tf.TensorSpec(shape=[None,4], dtype=tf.float32))
        # proposals = tf.vectorized_map(lambda p: utils.norm_boxes(p, tuple(self.config.IMAGE_SHAPE[:2])), proposals)
        
        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)