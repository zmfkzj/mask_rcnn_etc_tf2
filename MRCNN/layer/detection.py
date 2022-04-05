import tensorflow as tf
import tensorflow.keras.layers as KL

from MRCNN.config import Config
from .proposal import apply_box_deltas_graph, clip_boxes_graph
from ..model_utils.data_formatting import parse_image_meta_graph
from ..model_utils.miscellenous_graph import norm_boxes_graph
from MRCNN import utils

def refine_detections_graph(rois, probs, deltas, window, config):
    batch_size = tf.shape(rois)[0]
    class_ids = tf.argmax(probs, axis=2, output_type=tf.int32)
    # Class probability of the top class of each ROI
    _indices = tf.repeat(tf.expand_dims(tf.range(tf.shape(probs)[1]),axis=0),batch_size,axis=0)
    indices = tf.stack([_indices, class_ids], axis=2)
    class_scores = tf.gather_nd(probs, indices, batch_dims=1)
    # Class-specific bounding box deltas
    deltas_specific = tf.gather_nd(deltas, indices, batch_dims=1)
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = apply_box_deltas_graph(
        rois, deltas_specific * config.BBOX_STD_DEV)
    # Clip boxes to image window
    refined_rois = clip_boxes_graph(refined_rois, window)

    def _refine_detections_graph(inputs):
        """Refine classified proposals and filter overlaps and return final
        detections.

        Inputs:
            rois: [N, (y1, x1, y2, x2)] in normalized coordinates
            probs: [N, num_classes]. Class probabilities.
            deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                    bounding box deltas.
            window: (y1, x1, y2, x2) in normalized coordinates. The part of the image
                that contains the image excluding the padding.

        Returns detections shaped: [num_detections, (y1, x1, y2, x2, class_id, score)] where
            coordinates are normalized.
        """
        class_ids = inputs[0]
        class_scores = inputs[1]
        refined_rois = inputs[2]
        # Class IDs per ROI

        # TODO: Filter out boxes with zero area

        # Filter out background boxes
        keep = tf.where(class_ids > 0)[:, 0]
        # Filter out low confidence boxes
        if config.DETECTION_MIN_CONFIDENCE:
            conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
            keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                            tf.expand_dims(conf_keep, 0))
            keep = tf.sparse.to_dense(keep)[0]

        # Apply per-class NMS
        # 1. Prepare variables
        pre_nms_class_ids = tf.gather(class_ids, keep)
        pre_nms_scores = tf.gather(class_scores, keep)
        pre_nms_rois = tf.gather(refined_rois,   keep)
        unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

        def nms_keep_map(class_id):
            """Apply Non-Maximum Suppression on ROIs of the given class."""
            # Indices of ROIs of the given class
            ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
            # Apply NMS
            class_keep = tf.image.non_max_suppression(
                    tf.gather(pre_nms_rois, ixs),
                    tf.gather(pre_nms_scores, ixs),
                    max_output_size=config.DETECTION_MAX_INSTANCES,
                    iou_threshold=config.DETECTION_NMS_THRESHOLD)
            # Map indices
            class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
            # Pad with -1 so returned tensors have the same shape
            gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
            class_keep = tf.pad(class_keep, [(0, gap)],
                                mode='CONSTANT', constant_values=-1)
            # Set shape so map_fn() can infer result shape
            class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
            return class_keep

        # 2. Map over class IDs
        nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                            dtype=tf.int64)
        # 3. Merge results into one list, and remove -1 padding
        nms_keep = tf.reshape(nms_keep, [-1])
        nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
        # 4. Compute intersection between keep and nms_keep
        keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(nms_keep, 0))
        keep = tf.sparse.to_dense(keep)[0]
        # Keep top detections
        roi_count = config.DETECTION_MAX_INSTANCES
        class_scores_keep = tf.gather(class_scores, keep)
        num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
        top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
        keep = tf.gather(keep, top_ids)

        # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
        # Coordinates are normalized.
        detections = tf.concat([
            tf.gather(refined_rois, keep),
            tf.cast(tf.gather(class_ids, keep), tf.float32)[..., tf.newaxis],
            tf.gather(class_scores, keep)[..., tf.newaxis]
            ], axis=1)

        # Pad with zeros if detections < DETECTION_MAX_INSTANCES
        gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
        detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
        print(keep)
        return detections, keep
    detections, keep = tf.map_fn(_refine_detections_graph, 
                                [class_ids, class_scores, refined_rois], 
                                fn_output_signature=(tf.TensorSpec(shape=(config.DETECTION_MAX_INSTANCES, 6), dtype=tf.float32),
                                                    tf.TensorSpec(shape=(None,), dtype=tf.int64)))
    return detections, keep


class DetectionLayer(KL.Layer):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
    coordinates are normalized.
    """

    def __init__(self, config:Config=Config(), **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        image_meta = inputs[3]
        batch_size = tf.shape(rois)[0]
        # Get windows of images in normalized coordinates. Windows are the area
        # in the image that excludes the padding.
        # Use the shape of the first image in the batch to normalize the window
        # because we know that all images get resized to the same size.
        m = parse_image_meta_graph(image_meta)
        image_shape = m['image_shape'][0]
        window = norm_boxes_graph(m['window'], image_shape[:2])

        # Run detection refinement graph on each item in the batch
        detections_batch, keep_batch = refine_detections_graph(rois, mrcnn_class, mrcnn_bbox, window,self.config)
        # detections_batch = utils.batch_slice(
        #     [rois, mrcnn_class, mrcnn_bbox, window],
        #     lambda x, y, w, z: refine_detections_graph(x, y, w, z, self.config),
        #     batch_size)

        # Reshape output
        # [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] in
        # normalized coordinates
        return tf.reshape( detections_batch, [batch_size, self.config.DETECTION_MAX_INSTANCES, 6]),\
                keep_batch

    def compute_output_shape(self, input_shape):
        return (None, self.config.DETECTION_MAX_INSTANCES, 6)