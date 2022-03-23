import os
import random
import tensorflow as tf
import numpy as np
import skimage
import skimage.io

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
        image_pathes = []
        for r,_,fs in os.walk(image_dir):
            for f in fs:
                if Path(f).suffix.lower() in ['.jpg', '.jpeg','.png']:
                    full_path = Path(r)/f
                    image_pathes.append(str(full_path))
        pathes_size = len(image_pathes)
        if limit_step==-1:
            tqdm_total = int(np.ceil(pathes_size/self.config.BATCH_SIZE))
            take_count = pathes_size
        else:
            tqdm_total = limit_step
            take_count = limit_step*self.config.IMAGES_PER_GPU*self.config.GPU_COUNT

        all_pathes = []
        all_rois = []
        all_class_ids = []
        all_scores = []
        all_masks = []

        if shuffle:
            path_dataset = tf.data.Dataset.from_tensor_slices(image_pathes).shuffle(pathes_size).take(take_count).as_numpy_iterator()
        else:
            path_dataset = tf.data.Dataset.from_tensor_slices(image_pathes).take(take_count).as_numpy_iterator()

        mold_things = tf.data.Dataset.from_generator(lambda : self.gen_image(path_dataset),
                                                output_signature=(tf.TensorSpec(shape=(), dtype=tf.string),
                                                                tf.TensorSpec(shape=(3,), dtype=tf.int32),
                                                                tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
                                                                tf.TensorSpec(shape=(12+self.config.NUM_CLASSES,), dtype=tf.float32),
                                                                tf.TensorSpec(shape=(4,), dtype=tf.float32)))\
                                    .batch(self.config.BATCH_SIZE,num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

        pbar = tqdm(unit='step', total = tqdm_total)
        with self.mirrored_strategy.scope():
            mold_things = self.mirrored_strategy.experimental_distribute_dataset(mold_things)
            for dist_input in mold_things:
                pathes, image_shapes, detections, mrcnn_mask, molded_images, windows =self.detect_step(dist_input) 
                molded_shape = np.array([img.shape for img in molded_images.numpy()])
                for i in range(detections.numpy().shape[0]):
                    final_rois, final_class_ids, final_scores, final_masks =\
                        self.unmold_detections(detections[i].numpy(), mrcnn_mask[i].numpy(),image_shapes[i].numpy(), 
                                            molded_shape[i], windows[i].numpy())
                    
                    all_pathes.append(pathes[i].numpy())
                    all_rois.append(final_rois)
                    all_class_ids.append(final_class_ids)
                    all_scores.append(final_scores)
                    all_masks.append(final_masks)

                pbar.update()
        pbar.close()
        
        results = self.post_proceccing(all_pathes, all_rois, all_class_ids, all_scores, all_masks)
        for r in results:
            r['path'] = str(Path(r['path'].decode('utf-8')).relative_to(image_dir))
        return results
    
    @tf.function
    def detect_step(self, dist_input):
        def step_fn(pathes, image_shapes, molded_images, image_metas, windows):
            # Validate image sizes
            # All images in a batch MUST be of the same size
            image_shape = tf.shape(molded_images)[1:]
            for g in molded_images[1:]:
                tf.debugging.assert_equal(tf.shape(g), image_shape,"After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes.")

            # Run object detection
            detections, _, _, mrcnn_mask, _, _, _ =\
                self.model(molded_images, image_metas,training=False)
            
            return pathes, image_shapes, detections, mrcnn_mask, molded_images, windows

        outputs = self.mirrored_strategy.run(step_fn, args=[*dist_input])

        pathes, image_shapes, detections, mrcnn_mask, molded_images, windows = self.mirrored_strategy.gather(outputs,None)
        return pathes, image_shapes, detections, mrcnn_mask, molded_images, windows

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
            0, tf.shape(image), tf.shape(molded_image), window, scale,
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
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1)\
            if full_masks else np.empty(original_image_shape[:2] + (0,))

        return boxes, class_ids, scores, full_masks

    def gen_image(self, path_dataset):
        for path in path_dataset:
            image = skimage.io.imread(path.decode('utf-8'))
            # If grayscale. Convert to RGB for consistency.
            if image.ndim != 3:
                image = skimage.color.gray2rgb(image)
            # If has an alpha channel, remove it for consistency
            if image.shape[-1] == 4:
                image = image[..., :3]
            molded_image, image_meta, window = self.mold_inputs(image)
            
            yield path, image.shape, molded_image, image_meta, window

    def post_proceccing(self,all_pathes, all_rois, all_class_ids, all_scores, all_masks):

        results = []
        for i in range(len(all_pathes)):
            for j in range(all_rois[i].shape[0]):
                results.append({
                    "path": all_pathes[i],
                    "rois": all_rois[i][j], # x1,y1,x2,y2
                    "classes": self.classes[all_class_ids[i][j]],
                    "class_ids": all_class_ids[i][j],
                    "scores": all_scores[i][j],
                    "masks": all_masks[i][j],
                })
        return results