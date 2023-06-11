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
    @tf.function
    def call(self, x, counts, num_rows):
        """Picks different number of values from each row
        in x depending on the values in counts.
        """
        outputs = tf.zeros([tf.reduce_sum(counts),4],tf.float16)
        start = 0
        for i in tf.range(num_rows):
            stop = start + counts[i]
            indices = tf.reshape(tf.range(start, stop), [-1,1])
            outputs = tf.tensor_scatter_nd_update(outputs, indices, x[i, :counts[i]])
            start = stop
        return outputs
        # outputs = []
        # counts = tf.cast(counts, tf.int32)
        # for i in range(num_rows):
        #     outputs.append(x[i, :counts[i]])
        # return tf.concat(outputs, axis=0)


class NormBoxesGraph(KL.Layer):
    @tf.function
    def call(self, boxes, shape):
        """Converts boxes from pixel coordinates to normalized coordinates.
        boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
        shape: [..., (height, width)] in pixels

        Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
        coordinates it's inside the box.

        Returns:
            [..., (y1, x1, y2, x2)] in normalized coordinates
        """
        h, w = tf.split(tf.cast(shape, tf.float16), 2)
        scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0, dtype=tf.float16)
        shift = tf.constant([0., 0., 1., 1.], dtype=tf.float16)
        boxes = tf.cast(boxes, tf.float16)
        return tf.divide(boxes - shift, scale)


class DenormBoxesGraph(KL.Layer):
    @tf.function
    def call(self,boxes, shape):
        """Converts boxes from normalized coordinates to pixel coordinates.
        boxes: [..., (y1, x1, y2, x2)] in normalized coordinates
        shape: [..., (height, width)] in pixels

        Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
        coordinates it's inside the box.

        Returns:
            [..., (y1, x1, y2, x2)] in pixel coordinates
        """
        h, w = tf.split(tf.cast(shape, tf.float16), 2)
        scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0, dtype=tf.float16)
        shift = tf.constant([0., 0., 1., 1.], dtype=tf.float16)
        return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int16)