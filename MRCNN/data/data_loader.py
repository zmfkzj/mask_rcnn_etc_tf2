import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import imgaug.augmenters as iaa
import numpy as np
import tensorflow as tf
import cv2
from imgaug.augmentables.bbs import BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmenters import Sequential
from pycocotools import mask as maskUtils

from MRCNN.config import Config
from MRCNN.data.dataset import Dataset
from MRCNN.utils import (compute_backbone_shapes, compute_overlaps,
                         generate_pyramid_anchors)


class Mode(Enum):
    PREDICT = 'predict'
    TEST = 'test'
    TRAIN = 'train'


@dataclass
class DataLoader:
    config:Config
    mode:Mode
    batch_size:int
    active_class_ids:Optional[list[int]] = None
    dataset:Optional[Dataset] = None
    image_pathes:Optional[Union[str,list[str]]] = None
    augmentations:Optional[Sequential] = None

    def __post_init__(self):
        backbone_shapes = compute_backbone_shapes(self.config)
        self.anchors = generate_pyramid_anchors(
                            self.config.RPN_ANCHOR_SCALES,
                            self.config.RPN_ANCHOR_RATIOS,
                            backbone_shapes,
                            self.config.BACKBONE_STRIDES,
                            self.config.RPN_ANCHOR_STRIDE)

        if self.mode in [Mode.TRAIN, Mode.TEST]:
            coco = self.dataset.coco

        if self.mode == Mode.TRAIN:
            path = tf.data.Dataset\
                .from_tensor_slices([img['path'] for img in coco.dataset['images']])

            ann_ids = [self.padding_ann_ids(coco.getAnnIds(img['id'], self.active_class_ids)) 
                       for img in coco.dataset['images']]
            ann_ids = tf.data.Dataset.from_tensor_slices(ann_ids)

            self.data_loader = tf.data.Dataset\
                .zip((path, ann_ids))\
                .shuffle(len(self.dataset))\
                .map(lambda path, ann_ids: self.preproccessing_train(path, ann_ids), 
                     num_parallel_calls=tf.data.AUTOTUNE)\

            active_dataloader_class_ids = [0]+[self.dataset.get_dataloader_class_id(id) for id in self.active_class_ids]
            num_classes = len(coco.dataset['categories'])+1
            active_classes = [1 if dataloader_class_id in active_dataloader_class_ids else 0 for dataloader_class_id in range(num_classes)]

            active_classes_dataset = tf.data.Dataset\
                .from_tensors(active_classes)\
                .repeat()

            self.data_loader = tf.data.Dataset\
                .zip((self.data_loader, active_classes_dataset))\
                .map(lambda datas, active_class_id: [*datas, active_class_id],
                     num_parallel_calls=tf.data.AUTOTUNE)\
                .batch(self.batch_size)\
                .map(lambda *datas: [datas],
                     num_parallel_calls=tf.data.AUTOTUNE)\
                .repeat()\
                .prefetch(tf.data.AUTOTUNE)

        elif self.mode == Mode.PREDICT:
            if isinstance(self.image_pathes, str):
                self.image_pathes = [self.image_pathes]
            
            pathes = []
            for p in self.image_pathes:
                if os.path.isdir(p):
                    for r,_,fs in os.walk(p):
                        for f in fs:
                            if Path(f).suffix.lower() in ['.jpg', '.jpeg','.png']:
                                full_path = Path(r)/f
                                pathes.append(str(full_path))
                elif os.path.isfile(p):
                    if Path(p).suffix.lower() in ['.jpg', '.jpeg','.png']:
                        pathes.append(p)
                else:
                    raise FileNotFoundError(f'{p}를 찾을 수 없습니다.')
                
            self.data_loader = tf.data.Dataset\
                .from_tensor_slices(pathes)\
                .map(self.preprocessing_predict, num_parallel_calls=tf.data.AUTOTUNE)\
                .batch(self.batch_size)\
                .map(lambda *datas: [datas],
                     num_parallel_calls=tf.data.AUTOTUNE)\
                .prefetch(tf.data.AUTOTUNE)
        
        else:
            pathes = tf.data.Dataset\
                .from_tensor_slices([img['path'] for img in coco.dataset['images']])
            img_ids = tf.data.Dataset\
                .from_tensor_slices([img['id'] for img in coco.dataset['images']])


            self.data_loader = tf.data.Dataset\
                .zip((pathes,img_ids))\
                .shuffle(len(self.dataset))\
                .map(self.preproccessing_test, num_parallel_calls=tf.data.AUTOTUNE)\
                .batch(self.batch_size)\
                .map(lambda *datas: [datas],
                     num_parallel_calls=tf.data.AUTOTUNE)\
                .repeat()\
                .prefetch(tf.data.AUTOTUNE)


    def __iter__(self):
        return iter(self.data_loader)
    

    def __hash__(self) -> int:
        return hash((tuple(self.active_class_ids) if self.active_class_ids is not None else None, 
                     self.mode, 
                     tuple(self.image_pathes) if self.image_pathes is not None else None, 
                     self.dataset))


    @tf.function
    def preprocessing_predict(self, path):
        image = self.load_image(path)
        resized_image, window = self.resize_image(image)
        origin_image_shape = tf.shape(image)
        return resized_image, window, origin_image_shape, path
    
    @tf.function
    def preproccessing_test(self, path, img_id):
        image = self.load_image(path)
        resized_image, window = self.resize_image(image)
        origin_image_shape = tf.shape(image)
        return resized_image, window, origin_image_shape, img_id

    def load_gt(self, ann_ids, h, w):
        ann_ids = ann_ids.numpy()
        ann_ids = [ann_id for ann_id in ann_ids if ann_id!=0]
        anns = self.dataset.coco.loadAnns(ann_ids)

        gts = [gt for gt in [self.load_ann(ann, (h,w)) for ann in anns] if gt is not None]
        if gts:
            boxes, masks, dataloader_class_ids = list(zip(*gts))
            
            boxes = tf.cast(tf.stack(boxes), tf.float32)
            masks = tf.cast(tf.stack(masks), tf.bool)
            dataloader_class_ids = tf.cast(tf.stack(dataloader_class_ids), tf.int64)
        else:
            boxes = tf.zeros([0,4], dtype=tf.float32)
            masks = tf.zeros([0,h,w], dtype=tf.bool)
            dataloader_class_ids = tf.zeros([0], dtype=tf.int64)
        return boxes, masks, dataloader_class_ids

    @tf.function
    def preproccessing_train(self, path, ann_ids):
        image = self.load_image(path)
        boxes, masks, dataloader_class_ids =\
            tf.py_function(self.load_gt, (ann_ids,tf.shape(image)[0],tf.shape(image)[1]),(tf.float32, tf.bool, tf.int64))
    
        if self.augmentations is not None:
            image, boxes, masks = \
                self.augment(image, boxes, masks)

        resized_image, resized_boxes, minimize_masks = \
            self.resize(image, boxes, masks)
        rpn_match, rpn_bbox = \
            self.build_rpn_targets(dataloader_class_ids, resized_boxes)

        
        pooled_box = tf.zeros([self.config.MAX_GT_INSTANCES,4],dtype=tf.float32)
        pooled_mask = tf.zeros([self.config.MAX_GT_INSTANCES,*self.config.MINI_MASK_SHAPE],dtype=tf.bool)
        pooled_class_id = tf.zeros([self.config.MAX_GT_INSTANCES],dtype=tf.int64)

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

        return resized_image, resized_boxes, minimize_masks, dataloader_class_ids,rpn_match, rpn_bbox


    @tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.int64)])
    def padding_ann_ids(self, ann_ids):
        def f(ann_ids):
            shape = ann_ids[0].shape
            padded_ann_ids = tf.zeros([self.config.MAX_GT_INSTANCES, *shape], dtype=tf.int64)
            indices = tf.expand_dims(tf.range(tf.shape(ann_ids)[0]),1)
            ann_ids = tf.tensor_scatter_nd_update(padded_ann_ids, indices, tf.gather(ann_ids, tf.range(tf.shape(ann_ids)[0])))
            return ann_ids

        if tf.shape(ann_ids)[0]>self.config.MAX_GT_INSTANCES:
            ann_ids = tf.random.shuffle(ann_ids)[:self.config.MAX_GT_INSTANCES]
            ann_ids = f(ann_ids)
        else:
            ann_ids = tf.stack(ann_ids)
            ann_ids = f(ann_ids)
        
        return ann_ids


    @staticmethod
    @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)])
    def load_image(path):
        img_raw = tf.io.read_file(path)
        image = tf.io.decode_image(img_raw,3)
        image = tf.cond(tf.shape(image)[-1]==1,
                        lambda : tf.image.grayscale_to_rgb(image),
                        lambda : image)
        image = image[:,:,:3]
        if tf.size(image) == 0:
            tf.print(f"image({path}) size is zero")
        return image
    

    @tf.function
    def resize(self, image, bbox, mask):
        origin_shape = tf.shape(image)
        resized_image, window = self.resize_image(image)
        resized_bbox = self.resize_box(bbox, origin_shape, window)
        minimize_mask = self.minimize_mask(bbox, mask, self.config.MINI_MASK_SHAPE)
        return resized_image, resized_bbox, minimize_mask
        

    @tf.function
    def resize_image(self, image):
        """Resizes an image keeping the aspect ratio unchanged.

        min_dim: if provided, resizes the image such that it's smaller
            dimension == min_dim
        max_dim: if provided, ensures that the image longest side doesn't
            exceed this value.
        min_scale: if provided, ensure that the image is scaled up by at least
            this percent even if min_dim doesn't require it.

        Returns:
        image: the resized image
        window: (y1, x1, y2, x2). If max_dim is provided, padding might
            be inserted in the returned image. If so, this window is the
            coordinates of the image part of the full image (excluding
            the padding). The x2, y2 pixels are not included.
        scale: The scale factor used to resize the image
        padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
        """
        image = tf.ensure_shape(image, [None, None,3])
        image = tf.image.resize(image, 
                                list(self.config.IMAGE_SHAPE[:2]), 
                                preserve_aspect_ratio=True, 
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # Get new height and width
        h = tf.shape(image)[0]
        w = tf.shape(image)[1]
        top_pad = (self.config.IMAGE_MAX_DIM - h) // 2
        bottom_pad = self.config.IMAGE_MAX_DIM - h - top_pad
        left_pad = (self.config.IMAGE_MAX_DIM - w) // 2
        right_pad = self.config.IMAGE_MAX_DIM - w - left_pad
        padding = tf.stack([(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)])
        image = tf.pad(image, padding, mode='constant', constant_values=0)
        window = tf.stack((top_pad, left_pad, h + top_pad, w + left_pad))
        return image, window
    

    def load_ann(self, ann, image_shape):
        if (ann['category_id'] not in self.active_class_ids):
            return None

        dataloader_class_id = self.dataset.get_dataloader_class_id(ann['category_id'])

        x1,y1,w,h = ann['bbox']
        box = np.array((y1,x1,y1+h,x1+w))

        y1, x1, y2, x2 = np.round(box)
        area = (y2-y1)*(x2-x1)
        if area == 0:
            return None

        h = image_shape[0]
        w = image_shape[1]
        mask = self.annToMask(ann, h, w)

        if ann['iscrowd']:
            # Use negative class ID for crowds
            dataloader_class_id *= -1
            # For crowd masks, annToMask() sometimes returns a mask
            # smaller than the given dimensions. If so, resize it.

        return box, mask, dataloader_class_id


    @staticmethod
    @tf.function
    def resize_box(bboxes, origin_shape, window):
        origin_shape = tf.cast(origin_shape, tf.float32)
        window = tf.cast(window, tf.float32)
        o_h = origin_shape[0]
        o_w = origin_shape[1]
        y1 = window[0]
        x1 = window[1]
        y2 = window[2]
        x2 = window[3]
        n_h = y2-y1
        n_w = x2-x1
        scale = tf.stack([n_h,n_w,n_h,n_w])/tf.stack([o_h,o_w,o_h,o_w])
        new_box = bboxes * scale + tf.stack([y1,x1,y1,x1])
        return new_box


    @tf.function
    def padding_bbox(self, boxes, cls_ids):
        boxes = tf.pad(boxes,[[0,self.config.MAX_GT_INSTANCES-tf.shape(boxes)[0]],[0,0]])
        cls_ids = tf.reshape(cls_ids, [-1,1])
        cls_ids = tf.pad(cls_ids,[[0,self.config.MAX_GT_INSTANCES-tf.shape(cls_ids)[0]],[0,0]])
        cls_ids = tf.squeeze(cls_ids, 1)
        return boxes, cls_ids


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
                            (tf.uint8,tf.float32,tf.bool), name='augment')


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
            y1 = tf.cast(tf.round(b[0]), tf.int64)
            x1 = tf.cast(tf.round(b[1]), tf.int64)
            y2 = tf.cast(tf.round(b[2]), tf.int64)
            x2 = tf.cast(tf.round(b[3]), tf.int64)

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


    @tf.function
    def build_rpn_targets(self, gt_class_ids, gt_boxes):
        """Given the anchors and GT boxes, compute overlaps and identify positive
        anchors and deltas to refine them to match their corresponding GT boxes.

        anchors: [num_anchors, (y1, x1, y2, x2)]
        gt_class_ids: [num_gt_boxes] Integer class IDs.
        gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

        Returns:
        rpn_match: [N] (int32) matches between anchors and GT boxes.
                1 = positive anchor, -1 = negative anchor, 0 = neutral
        rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
        """
        # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
        rpn_match = tf.zeros([self.anchors.shape[0]], dtype=tf.int32)
        # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
        rpn_bbox = tf.TensorArray(tf.float32, size=self.config.RPN_TRAIN_ANCHORS_PER_IMAGE)

        if tf.shape(gt_boxes)[0] == 0:
            rpn_bbox = tf.zeros([self.config.RPN_TRAIN_ANCHORS_PER_IMAGE,4], dtype=tf.float32)
            return tf.cast(rpn_match,tf.int64), rpn_bbox

        # Handle COCO crowds
        # A crowd box in COCO is a bounding box around several instances. Exclude
        # them from training. A crowd box is given a negative class ID.
        crowd_ix = tf.squeeze(tf.where(gt_class_ids < 0),-1)
        if tf.shape(crowd_ix)[0] > 0:
            # Filter out crowds from ground truth class IDs and boxes
            non_crowd_ix = tf.squeeze(tf.where(gt_class_ids > 0),-1)
            crowd_boxes = tf.gather(gt_boxes,crowd_ix)
            gt_class_ids = tf.gather(gt_class_ids,non_crowd_ix)
            gt_boxes = tf.gather(gt_boxes,non_crowd_ix)
            # Compute overlaps with crowd boxes [anchors, crowds]
            crowd_overlaps = compute_overlaps(self.anchors, crowd_boxes)
            crowd_iou_max = tf.cast(tf.reduce_max(crowd_overlaps, axis=1),tf.float32)
            no_crowd_bool = (crowd_iou_max < 0.001)
        else:
            # All anchors don't intersect a crowd
            no_crowd_bool = tf.ones([self.anchors.shape[0]], dtype=tf.bool)
        
        # Compute overlaps [num_anchors, num_gt_boxes]
        overlaps = compute_overlaps(self.anchors, gt_boxes)

        # Match anchors to GT Boxes
        # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
        # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
        # Neutral anchors are those that don't match the conditions above,
        # and they don't influence the loss function.
        # However, don't keep any GT box unmatched (rare, but happens). Instead,
        # match it to the closest anchor (even if its max IoU is < 0.3).
        #
        # 1. Set negative anchors first. They get overwritten below if a GT box is
        # matched to them. Skip boxes in crowd areas.
        anchor_iou_argmax = tf.argmax(overlaps, axis=1, output_type=tf.int32)
        
        indices = tf.transpose([tf.range(tf.shape(overlaps)[0]), anchor_iou_argmax])
        anchor_iou_max = tf.gather_nd(overlaps,indices)
        anchor_iou_max = tf.ensure_shape(anchor_iou_max, [self.anchors.shape[0]])
        no_crowd_bool = tf.ensure_shape(no_crowd_bool, [self.anchors.shape[0]])
        rpn_match = tf.where((anchor_iou_max < 0.3) & no_crowd_bool, -1, rpn_match)
        # 2. Set an anchor for each GT box (regardless of IoU value).
        # If multiple anchors have the same IoU match all of them
        rpn_match = tf.where(tf.reduce_any(overlaps == tf.reduce_max(overlaps, axis=0), axis=1), 1, rpn_match)

        # 3. Set anchors with high overlap as positive.
        rpn_match = tf.where(anchor_iou_max >= 0.7, 1, rpn_match) 

        # Subsample to balance positive and negative anchors
        # Don't let positives be more than half the anchors
        ids = tf.transpose(tf.where(rpn_match == 1))[0]
        extra = tf.shape(ids)[0] - (self.config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
        if extra > 0:
            # Reset the extra ones to neutral
            ids = tf.random.shuffle(ids)[:extra]
            rpn_match = tf.tensor_scatter_nd_update(rpn_match, 
                                                    tf.expand_dims(ids,1),
                                                    tf.zeros([tf.shape(ids)[0]],tf.int32))

        # Same for negative proposals
        ids = tf.transpose(tf.where(rpn_match == -1))[0]
        extra = tf.shape(ids)[0] - (self.config.RPN_TRAIN_ANCHORS_PER_IMAGE - tf.reduce_sum(tf.cast(rpn_match == 1,tf.int32)))
        if extra > 0:
            # Rest the extra ones to neutral
            ids = tf.random.shuffle(ids)[:extra]
            rpn_match = tf.tensor_scatter_nd_update(rpn_match, 
                                                    tf.expand_dims(ids,1),
                                                    tf.zeros([tf.shape(ids)[0]], tf.int32))

        # For positive anchors, compute shift and scale needed to transform them
        # to match the corresponding GT boxes.
        ids = tf.transpose(tf.where(rpn_match == 1))[0]
        ix = 0  # index into rpn_bbox
        # TODO: use box_refinement() rather than duplicating the code here
        gathered_anchores = tf.cast(tf.gather(self.anchors,ids), tf.float32)

        for i in tf.range(tf.shape(ids)[0]):
        # for i, a in zip(ids, tf.gather(self.anchors,ids)):
            # Closest gt box (it might have IoU < 0.7)
            gt = gt_boxes[anchor_iou_argmax[ids[i]]]

            # Convert coordinates to center plus width/height.
            # GT Box
            gt_h = gt[2] - gt[0]
            gt_w = gt[3] - gt[1]
            gt_center_y = gt[0] + 0.5 * gt_h
            gt_center_x = gt[1] + 0.5 * gt_w
            # Anchor
            a_h = gathered_anchores[i][2] - gathered_anchores[i][0]
            a_w = gathered_anchores[i][3] - gathered_anchores[i][1]
            a_center_y = gathered_anchores[i][0] + 0.5 * a_h
            a_center_x = gathered_anchores[i][1] + 0.5 * a_w

            # Compute the bbox refinement that the RPN should predict.
            rpn_bbox = rpn_bbox.write(ix, tf.stack([(gt_center_y - a_center_y) / a_h, 
                                                        (gt_center_x - a_center_x) / a_w,
                                                        tf.math.log(gt_h / a_h), 
                                                        tf.math.log(gt_w / a_w),
                                                        ])/self.config.RPN_BBOX_STD_DEV)
            ix += 1
        rpn_bbox = rpn_bbox.stack()

        return tf.cast(rpn_match, tf.int64), rpn_bbox