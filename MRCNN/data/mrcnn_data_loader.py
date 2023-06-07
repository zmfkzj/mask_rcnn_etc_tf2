import numpy as np
import tensorflow as tf
from imgaug.augmentables.bbs import BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from pycocotools import mask as maskUtils

from MRCNN.data import frcnn_data_loader


class DataLoader(frcnn_data_loader.DataLoader):
    
    def load_gt(self, ann_ids, h, w):
        ann_ids = ann_ids.numpy()
        ann_ids = [ann_id for ann_id in ann_ids if ann_id!=0]
        anns = self.dataset.coco.loadAnns(ann_ids)

        gts = [gt for gt in [self.load_ann(ann, (h,w)) for ann in anns] if gt is not None]
        if gts:
            boxes, masks, dataloader_class_ids = tuple(zip(*gts))
            
            boxes = tf.cast(tf.stack(boxes), tf.float16)
            masks = tf.cast(tf.stack(masks), tf.bool)
            dataloader_class_ids = tf.cast(tf.stack(dataloader_class_ids), tf.int16)
        else:
            boxes = tf.zeros([0,4], dtype=tf.float16)
            masks = tf.zeros([0,h,w], dtype=tf.bool)
            dataloader_class_ids = tf.zeros([0], dtype=tf.int16)
        return boxes, masks, dataloader_class_ids

    @tf.function
    def preprocessing_train(self, path, ann_ids):
        image = self.load_image(path)
        boxes, masks, dataloader_class_ids =\
            tf.py_function(self.load_gt, (ann_ids,tf.shape(image)[0],tf.shape(image)[1]),(tf.float16, tf.bool, tf.int16))
    
        if self.augmentations is not None:
            image, boxes, masks = \
                self.augment(image, boxes, masks)

        resized_image, resized_boxes, minimize_masks = \
            self.resize(image, boxes, masks)

        preprocessed_image = self.config.PREPROCESSING(tf.cast(resized_image, tf.float16))

        rpn_match, rpn_bbox = \
            self.build_rpn_targets(dataloader_class_ids, resized_boxes)

        
        pooled_box = tf.zeros([self.config.MAX_GT_INSTANCES,4],dtype=tf.float16)
        pooled_mask = tf.zeros([self.config.MAX_GT_INSTANCES,*self.config.MINI_MASK_SHAPE],dtype=tf.bool)
        pooled_class_id = tf.zeros([self.config.MAX_GT_INSTANCES],dtype=tf.int16)

        instance_count = tf.shape(boxes)[0]
        if instance_count>self.config.MAX_GT_INSTANCES:
            indices = tf.random.shuffle(tf.range(instance_count))[:self.config.MAX_GT_INSTANCES]
            indices = tf.expand_dims(indices,1)
        else:
            indices = tf.range(instance_count)
            indices = tf.expand_dims(indices,1)

        resized_boxes = tf.tensor_scatter_nd_update(pooled_box, indices, tf.gather(resized_boxes, tf.squeeze(indices,-1)))
        minimize_masks = tf.tensor_scatter_nd_update(pooled_mask, indices, tf.gather(minimize_masks, tf.squeeze(indices,-1)))
        dataloader_class_ids = tf.tensor_scatter_nd_update(pooled_class_id, indices, tf.gather(dataloader_class_ids, tf.squeeze(indices, -1)))

        minimize_masks = tf.transpose(minimize_masks, [1,2,0])

        return {'input_images': preprocessed_image,
                'input_gt_boxes': resized_boxes,
                'dataloader_class_ids':dataloader_class_ids,
                'rpn_match':rpn_match,
                'rpn_bbox':rpn_bbox,
                'input_gt_masks':minimize_masks}


    @tf.function
    def resize(self, image, bbox, mask):
        resized_image, resized_bbox = super().resize(image, bbox)
        minimize_mask = self.minimize_mask(bbox, mask, self.config.MINI_MASK_SHAPE)
        return resized_image, resized_bbox, minimize_mask
        

    def load_ann(self, ann, image_shape):
        box, dataloader_class_id = super().load_ann(ann, image_shape)
        h = image_shape[0]
        w = image_shape[1]
        mask = self.annToMask(ann, h, w)
        return box, mask, dataloader_class_id


    @tf.function
    def augment(self, image, bbox, masks):
        def _augment(image,bbox,masks):
            """
            Args:
                image (_type_): (H,W,C)
                bbox (_type_): (Num_Object, (x1,y1,x2,y2))
                masks (_type_): (Num_Object,H,W)

            Returns:
                _type_: _description_
            """
            image = image.numpy()
            bbox = bbox.numpy()
            masks = masks.numpy()
            shape = image.shape[:2]

            bbox = bbox[:,[1,0,3,2]]
            bbi = BoundingBoxesOnImage.from_xyxy_array(bbox, shape)

            if masks.shape[0]>0:
                masks = masks.transpose([1,2,0])
                masks = SegmentationMapsOnImage(masks,tuple(shape))
                image, bbox, masks = self.augmentations(image=image, bounding_boxes=bbi, segmentation_maps=masks)
                bbox = bbox.to_xyxy_array()[:,[1,0,3,2]]
                masks = masks.get_arr().transpose([2,0,1])
            else:
                image, bbox = self.augmentations(image=image, bounding_boxes=bbi)
                bbox = bbox.to_xyxy_array()[:,[1,0,3,2]]

            return image,bbox,masks
        
        return tf.py_function(_augment, 
                            [image,bbox,masks],
                            (tf.uint8,tf.float16,tf.bool), name='augment')


    @staticmethod
    @tf.function
    def minimize_mask(bbox, mask, mini_shape):
        """Resize masks to a smaller version to reduce memory load.
        Mini-masks can be resized back to image scale using expand_masks()

        See inspect_data.ipynb notebook for more details.
        """
        def f(arg):
            m, b = arg
            m = tf.ensure_shape(m, [None,None])
            m = tf.cast(m,tf.bool)
            y1 = tf.cast(tf.round(b[0]), tf.int16)
            x1 = tf.cast(tf.round(b[1]), tf.int16)
            y2 = tf.cast(tf.round(b[2]), tf.int16)
            x2 = tf.cast(tf.round(b[3]), tf.int16)

            m = m[y1:y2, x1:x2]
            if tf.size(m) == 0:
                tf.print("mask size is zero")
                tf.errors.INVALID_ARGUMENT

            # Resize with bilinear interpolation
            m = tf.expand_dims(m,2)
            m = tf.cast(m, tf.uint8)
            m = tf.image.resize(m, mini_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            m = tf.squeeze(m, -1)
            m = tf.cast(m, tf.bool)
            return m
        
        mini_mask = tf.map_fn(f, [mask,bbox],fn_output_signature=tf.TensorSpec(shape=mini_shape, dtype=tf.bool))
        return mini_mask


    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle


    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m