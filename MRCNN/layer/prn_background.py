import keras.api._v2.keras.layers as KL
import tensorflow as tf

from MRCNN.config import Config


class PrnBackground(KL.Layer):
    def __init__(self, config:Config, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.config = config

    def call(self, input_images, gt_boxes, input_prn_images):
        gt_boxes = tf.cast(tf.math.round(gt_boxes), tf.int32)
        prn_batch_size = tf.shape(input_prn_images)[0]

        output_bg_prn_images = tf.TensorArray(tf.float32, size=prn_batch_size, element_shape=(*self.config.PRN_IMAGE_SIZE,4))

        for i in tf.range(tf.shape(input_images)[0]):
            if i==prn_batch_size:
                break
            img = input_images[i]
            boxes = gt_boxes[i]
            mask = tf.ones(self.config.IMAGE_SHAPE[:2])

            for box in boxes:
                if tf.reduce_all(box==0):
                    continue
                y1 = box[0]
                x1 = box[1]
                y2 = box[2]
                x2 = box[3]
                ys, xs = tf.meshgrid(tf.range(y1,y2), tf.range(x1,x2))
                ys = tf.reshape(ys, [-1])
                xs = tf.reshape(xs, [-1])
                indices = tf.transpose((ys,xs))
                updates = tf.zeros_like(indices, dtype=tf.float32)[:,0]
                mask = tf.tensor_scatter_nd_update(mask, indices, updates)
            
            mask = tf.expand_dims(mask, -1)
            bg_prn_image = tf.concat([img, mask], -1)
            bg_prn_image = tf.image.resize(bg_prn_image, self.config.PRN_IMAGE_SIZE)

            output_bg_prn_images.write(i, bg_prn_image)

        output_bg_prn_images = output_bg_prn_images.stack()
        output_bg_prn_images = tf.expand_dims(output_bg_prn_images, 1)
        output_prn_images = tf.concat([output_bg_prn_images, input_prn_images], 1)

        return output_prn_images
