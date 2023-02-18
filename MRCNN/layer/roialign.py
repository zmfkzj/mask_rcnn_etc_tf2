import numpy as np
import tensorflow as tf
import keras.api._v2.keras.layers as KL

from MRCNN.config import Config
from ..model_utils.data_formatting import parse_image_meta_graph

@tf.function
def log2_graph(x):
    """Implementation of Log2. TF doesn't have a native implementation."""
    return tf.math.log(x) / tf.math.log(2.0)


@tf.function
def pool_feature_maps(feature_maps, roi_level, boxes, pool_shape):
    # Loop through levels and apply ROI pooling to each. P2 to P5.
    pooled = tf.TensorArray(tf.float32, size=len(feature_maps))
    box_to_level = tf.TensorArray(tf.int32, size=len(feature_maps))
    i = 0
    for feature in feature_maps:
    # for i, level in enumerate(range(2, 2+len(feature_maps))):
        ix = tf.where(tf.equal(roi_level, i+2))
        level_boxes = tf.gather_nd(boxes, ix)

        # Box indices for crop_and_resize.
        box_indices = tf.cast(ix[:, 0], tf.int32)

        # Keep track of which box is mapped to which level
        box_to_level.write(i, tf.cast(ix, tf.int32))

        # Stop gradient propogation to ROI proposals
        level_boxes = tf.stop_gradient(level_boxes)
        box_indices = tf.stop_gradient(box_indices)

        # Crop and Resize
        # From Mask R-CNN paper: "We sample four regular locations, so
        # that we can evaluate either max or average pooling. In fact,
        # interpolating only a single value at each bin center (without
        # pooling) is nearly as effective."
        #
        # Here we use the simplified approach of a single value per bin,
        # which is how it's done in tf.crop_and_resize()
        # Result: [batch * num_boxes, pool_height, pool_width, channels]
        pooled.write(
            i,
            tf.image.crop_and_resize(**{'image':feature, 'boxes':level_boxes, 'box_indices':box_indices, 'crop_size':pool_shape, 'method':"bilinear"}))
        i+=1

    # Pack pooled features into one tensor
    pooled = pooled.concat()

    # Pack box_to_level mapping into one array and add another
    # column representing the order of pooled boxes
    box_to_level = box_to_level.concat()
    return pooled, box_to_level


class PyramidROIAlign(KL.Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - feature_maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    def __init__(self, pool_shape, config:Config, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)
        self.config = config

    def call(self, 
             boxes, # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
             image_shape, 
             feature_maps):# Feature Maps. List of feature maps from different level of the feature pyramid. Each is [batch, height, width, channels]
        
        
        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(tf.cast(2,tf.int64), tf.maximum(
            tf.cast(2, tf.int64), tf.cast(4, tf.int64) + tf.cast(tf.round(roi_level), tf.int64)))
        roi_level = tf.squeeze(roi_level, 2)

        pooled, box_to_level = pool_feature_maps(feature_maps, roi_level, boxes, self.pool_shape)

        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        pooled = tf.ensure_shape(pooled, [None, None, self.pool_shape[0], self.pool_shape[1], self.config.TOP_DOWN_PYRAMID_SIZE])
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )