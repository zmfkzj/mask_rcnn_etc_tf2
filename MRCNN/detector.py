import os
import random
import tensorflow as tf
import numpy as np

from MRCNN import utils
from MRCNN.config import Config
from MRCNN.model_utils.data_formatting import mold_image, compose_image_meta

from tqdm import tqdm
from pathlib import Path

class Detector:
    def __init__(self, model, classes:list, config:Config=Config(), ) -> None:
        self.mirrored_strategy = config.MIRRORED_STRATEGY
        self.config = config
        self.classes = classes

        with self.mirrored_strategy.scope():
            self.model = model
    
    def detect(self, image_dir, shuffle=False, limit_step=-1):
        """Runs the detection pipeline.

        image_pathes: List of pathes, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        image_pathes = []
        for r,_,fs in os.walk(image_dir):
            for f in fs:
                if Path(f).suffix.lower() in ['.jpg', '.jpeg','.png']:
                    full_path = Path(r)/f
                    image_pathes.append(str(full_path))

        things_gen = self.image_generator(image_pathes,shuffle)
        mold_things = tf.data.Dataset.from_generator(lambda : things_gen, output_types=(tf.string, tf.int32, tf.float32, tf.float64, tf.int32))\
                                    .batch(self.config.BATCH_SIZE).take(limit_step).prefetch(tf.data.AUTOTUNE)

        results = []
        pbar = tqdm(unit='step', total = limit_step)
        with self.mirrored_strategy.scope():
            mold_things = self.mirrored_strategy.experimental_distribute_dataset(mold_things)
            for dist_things in mold_things:
                pathes, image_shapes, molded_images, image_metas, windows = dist_things
                batch_detections, batch_mrcnn_mask = self.batch_detect(dist_things)

                #post-procecc
                post_detections = self.mirrored_strategy.run(self.post_proceccing, 
                                                            args=(pathes, image_shapes, batch_detections, batch_mrcnn_mask, molded_images ,windows))
                post_detections = self.mirrored_strategy.gather(post_detections, axis=None)
                results.extend(post_detections)

                pbar.update()

        for r in results:
            r['path'] = str(Path(r['path']).relative_to(image_dir))
        return results

    @tf.function
    def batch_detect(self, dist_things):
        batch_detections = self.mirrored_strategy.run(self.step_fn, args=(*dist_things,))
        # batch_detections = self.mirrored_strategy.gather(per_detections, axis=None)
        return batch_detections

    def step_fn(self, pathes, image_shapes, molded_images, image_metas, windows):
        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape,\
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        # Anchors
        anchors = self.model.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = tf.broadcast_to(anchors, (self.config.IMAGES_PER_GPU,) + anchors.shape)

        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _ =\
            self.model(molded_images, image_metas, anchors, training=False)
        return detections, mrcnn_mask

    def mold_inputs(self, image):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matrices [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matrices:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        # Resize image
        # TODO: move resizing to mold_image()
        molded_image, window, scale, padding, crop = utils.resize_image(
            image,
            min_dim=self.config.IMAGE_MIN_DIM,
            min_scale=self.config.IMAGE_MIN_SCALE,
            max_dim=self.config.IMAGE_MAX_DIM,
            mode=self.config.IMAGE_RESIZE_MODE)
        molded_image = mold_image(molded_image, self.config)
        # Build image_meta
        image_meta = compose_image_meta(
            0, image.shape, molded_image.shape, window, scale,
            np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
        return molded_image, image_meta, window

    def unmold_detections(self, detections, mrcnn_mask, original_image_shape,
                          image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
        mrcnn_mask: [N, height, width, num_classes]
        original_image_shape: [H, W, C] Original image shape before resizing
        image_shape: [H, W, C] Shape of the image after resizing and padding
        window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
                image is excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        # detections = tf.make_ndarray(tf.make_tensor_proto(detections))
        zero_ix = tf.where(detections[:, 4] == 0)[:,0]
        N = tf.cast(zero_ix[0], tf.int32) if tf.shape(zero_ix)[0] > 0 else tf.shape(detections)[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = tf.cast(detections[:N, 4],np.int32)
        scores = detections[:N, 5]
        mrcnn_mask = tf.transpose(mrcnn_mask,[0,3,1,2])
        indices = tf.transpose([tf.range(N),class_ids])
        masks = tf.gather_nd(mrcnn_mask,indices)

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = utils.norm_boxes(window, image_shape[:2])
        wy1 = window[0]
        wx1 = window[1]
        wy2 = window[2]
        wx2 = window[3]
        shift = tf.stack([wy1, wx1, wy1, wx1])
        # shift = tf.gather(window, [0,1,0,1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = tf.stack([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = tf.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        include_ix = tf.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) > 0)[:,0]
        if tf.shape(include_ix)[0] > 0:
            boxes = tf.gather(boxes, include_ix, axis=0)
            class_ids = tf.gather(class_ids, include_ix, axis=0)
            scores = tf.gather(scores, include_ix, axis=0)
            masks = tf.gather(masks, include_ix, axis=0)
            N = tf.shape(class_ids)[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
        full_masks = tf.stack(full_masks, axis=-1)\
            if full_masks else tf.zeros(tf.concat([original_image_shape[:2],0]),tf.bool)

        return boxes, class_ids, scores, full_masks

    def image_generator(self, pathes, shuffle=False):
        pathes = pathes[:]
        if shuffle:
            random.shuffle(pathes)

        for path in pathes:
            raw = tf.io.read_file(path)
            image = tf.io.decode_image(raw, channels=3)

            # Mold inputs to format expected by the neural network
            molded_image, image_meta, window = self.mold_inputs(image)
            yield path, image.shape, molded_image, image_meta, window

    def post_proceccing(self,pathes, image_shapes, detections, mrcnn_mask, molded_images ,windows):
        results = []
        for i in range(len(pathes)):
            path = pathes[i]
            shape = image_shapes[i]
            final_rois, final_class_ids, final_scores, final_masks =\
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                    shape, tf.shape(molded_images[i]),
                                    windows[i])
            results.append({
                "path": path.numpy().decode('utf-8'),
                "rois": final_rois.numpy(), # x1,y1,x2,y2
                "classes": [self.classes[id] for id in final_class_ids.numpy()],
                "class_ids": final_class_ids.numpy(),
                "scores": final_scores.numpy(),
                "masks": final_masks.numpy().astype(np.uint8),
            })
        return results