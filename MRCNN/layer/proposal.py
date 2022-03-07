import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as KL
from MRCNN import utils

def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)] boxes to update
    deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
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
    result = tf.stack([y1, x1, y2, x2], axis=2, name="apply_box_deltas_out")
    return result


def clip_boxes_graph(boxes, window):
    """
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] in the form y1, x1, y2, x2
    """
    # Split
    wy1, wx1, wy2, wx2 = tf.cond(tf.shape(tf.shape(window))==1,
                                 lambda : tf.split(tf.reshape(window,[1,tf.shape(window)[0]]),4,axis=1),
                                 lambda : tf.split(tf.broadcast_to(tf.expand_dims(window,1),[tf.shape(window)[0],1000,4]),4,axis=2))
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=2, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], clipped.shape[1], 4))
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

    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def call(self, inputs):
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]
        # Box deltas [batch, num_rois, 4]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
        # Anchors
        anchors = inputs[2]

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                         name="top_anchors").indices
        scores = tf.gather(scores, ix, axis=1,batch_dims=1)
        deltas = tf.gather(deltas, ix, axis=1,batch_dims=1)
        pre_nms_anchors = tf.gather(anchors,ix,axis=1,batch_dims=1, name="pre_nms_anchors")

        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, y2, x2)]
        boxes = apply_box_deltas_graph(pre_nms_anchors,deltas)

        # Clip to image boundaries. Since we're in normalized coordinates,
        # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = clip_boxes_graph(boxes, window)
        # boxes = utils.batch_slice(boxes,
        #                           lambda x: clip_boxes_graph(x, window),
        #                           batch_size,
        #                           names=["refined_anchors_clipped"])

        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy
        # for small objects, so we're skipping it.

        # Non-max suppression
        def nms(inputs):
            boxes = inputs[0]
            scores = inputs[1]
            indices = tf.image.non_max_suppression(
                boxes, scores, self.proposal_count,
                self.nms_threshold, name="rpn_non_max_suppression")
            proposals = tf.gather(boxes, indices)
            # Pad if needed
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals
        proposals = tf.map_fn(nms,[boxes, scores], dtype=tf.float32)
        
        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)