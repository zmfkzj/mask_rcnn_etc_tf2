import tensorflow as tf
import keras.api._v2.keras.layers as KL

@tf.function
def trim_zeros_graph(boxes, name='trim_zeros'):
    """Often boxes are represented with matrices of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


class BatchPackGraph(KL.Layer):
    def call(self, x, counts, num_rows):
        """Picks different number of values from each row
        in x depending on the values in counts.
        """
        shape = x[0, :counts[0]].shape
        outputs = tf.zeros([num_rows, *shape], dtype=tf.float32)
        indices = tf.expand_dims(tf.range(num_rows),1)
        outputs = tf.tensor_scatter_nd_update(outputs, indices, tf.gather(x, tf.range(num_rows))[:, :counts[0]])
        return outputs


class NormBoxesGraph(KL.Layer):
    def call(self, boxes, shape):
        """Converts boxes from pixel coordinates to normalized coordinates.
        boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
        shape: [..., (height, width)] in pixels

        Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
        coordinates it's inside the box.

        Returns:
            [..., (y1, x1, y2, x2)] in normalized coordinates
        """
        h, w = tf.split(tf.cast(shape, tf.float32), 2)
        scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
        shift = tf.constant([0., 0., 1., 1.])
        return tf.divide(boxes - shift, scale)

@tf.function
def denorm_boxes_graph(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [..., (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)

