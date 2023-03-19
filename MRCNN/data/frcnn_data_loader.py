import os
from pydantic.dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from official.vision.ops.iou_similarity import iou

import numpy as np
import tensorflow as tf
from imgaug.augmentables.bbs import BoundingBoxesOnImage
from imgaug.augmenters import Sequential

from MRCNN.config import Config
from MRCNN.data.input_datas import InputDatas
from MRCNN.enums import Mode
from MRCNN.data.dataset import Dataset
from MRCNN.utils import (compute_backbone_shapes,
                         generate_pyramid_anchors)


@dataclass
class DataLoader:
    config:Config
    mode:Mode
    dataset:Optional[Dataset] = None
    image_pathes:Optional[Union[str,list[str]]] = None
    augmentations:Optional[Sequential] = None
    batch_size:Optional[int] = None

    def __post_init__(self):
        backbone_shapes = compute_backbone_shapes(self.config)
        self.anchors = generate_pyramid_anchors(
                            self.config.RPN_ANCHOR_SCALES,
                            self.config.RPN_ANCHOR_RATIOS,
                            backbone_shapes,
                            self.config.BACKBONE_STRIDES,
                            self.config.RPN_ANCHOR_STRIDE)
        
        self.make_dataloader()
    

    def make_dataloader(self):
        if self.mode == Mode.PREDICT:
            self.data_loader = self.make_predict_dataloader()
        elif self.mode == Mode.TEST:
            self.data_loader = self.make_test_dataloader()
        elif self.mode == Mode.TRAIN:
            self.data_loader = self.make_train_dataloader()
        else:
            ValueError


    def make_predict_dataloader(self):
        self.batch_size = self.config.TEST_BATCH_SIZE if self.batch_size is None else self.batch_size

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
            
        data_loader = tf.data.Dataset\
            .from_tensor_slices(pathes)\
            .map(lambda p: self.preprocessing_predict(p), num_parallel_calls=tf.data.AUTOTUNE)\
            .batch(self.batch_size)\
            .map(lambda datas: 
                    [InputDatas(self.config, self.anchors.shape[0], **datas).to_dict()],
                    num_parallel_calls=tf.data.AUTOTUNE)\
            .prefetch(tf.data.AUTOTUNE)
        return data_loader
    

    def make_test_dataloader(self):
        coco = self.dataset.coco
        self.batch_size = self.config.TEST_BATCH_SIZE if self.batch_size is None else self.batch_size

        self.set_active_class_ids()

        pathes = tf.data.Dataset\
            .from_tensor_slices([img['path'] for img in coco.dataset['images']])
        img_ids = tf.data.Dataset\
            .from_tensor_slices([img['id'] for img in coco.dataset['images']])

        data_loader = tf.data.Dataset\
            .zip((pathes,img_ids))\
            .shuffle(len(self.dataset))\
            .map(self.preprocessing_test, num_parallel_calls=tf.data.AUTOTUNE)\
            .batch(self.batch_size)\
            .map(lambda datas: 
                    [InputDatas(self.config, self.anchors.shape[0], **datas).to_dict()],
                    num_parallel_calls=tf.data.AUTOTUNE)\
            .repeat()\
            .prefetch(tf.data.AUTOTUNE)
        return data_loader


    def make_train_dataloader(self):
        coco = self.dataset.coco
        self.batch_size = self.config.TRAIN_BATCH_SIZE if self.batch_size is None else self.batch_size

        self.set_active_class_ids()

        path = tf.data.Dataset\
            .from_tensor_slices([img['path'] for img in coco.dataset['images']])

        ann_ids = [self.padding_ann_ids(coco.getAnnIds(img['id'], self.active_class_ids)) 
                    for img in coco.dataset['images']]
        ann_ids = tf.data.Dataset.from_tensor_slices(ann_ids)

        data_loader = tf.data.Dataset\
            .zip((path, ann_ids))\
            .repeat()\
            .shuffle(len(self.dataset))\
            .map(lambda path, ann_ids: self.preprocessing_train(path, ann_ids), 
                    num_parallel_calls=tf.data.AUTOTUNE)\

        active_classes = [1]+[1 if cat in self.active_class_ids else 0 
                                for cat in self.active_class_ids]

        active_classes_dataset = tf.data.Dataset\
            .from_tensors(active_classes)\
            .repeat()

        data_loader = tf.data.Dataset\
            .zip((data_loader, active_classes_dataset))\
            .batch(self.batch_size)\
            .map(lambda datas, active_class_id: 
                 [InputDatas(self.config, 
                             self.anchors.shape[0], 
                             **datas).update_active_class_ids(active_class_id).to_dict()])\
            .prefetch(tf.data.AUTOTUNE)
        return data_loader


    def set_active_class_ids(self):
        if self.config.ACTIVE_CLASS_IDS is None:
            self.active_class_ids = [cat for cat in self.dataset.coco.cats]
        else:
            self.active_class_ids = self.config.ACTIVE_CLASS_IDS
        return self.active_class_ids


    def __iter__(self):
        return iter(self.data_loader)
    

    def __hash__(self) -> int:
        return hash((tuple(self.config.ACTIVE_CLASS_IDS) if self.config.ACTIVE_CLASS_IDS is not None else None, 
                     self.mode, 
                     tuple(self.image_pathes) if self.image_pathes is not None else None, 
                     self.dataset))

    @tf.function
    def preprocessing_predict(self, path):
        image = self.load_image(path)
        resized_image, window = self.resize_image(image, list(self.config.IMAGE_SHAPE[:2]))
        preprocessed_image = self.config.PREPROCESSING(tf.cast(resized_image, tf.float32))
        origin_image_shape = tf.shape(image)
        return {'input_images': preprocessed_image,
                'input_window': window,
                'origin_image_shapes':origin_image_shape,
                'pathes':path}
    
    # @tf.function
    def preprocessing_test(self, path, img_id):
        image = self.load_image(path)
        resized_image, window = self.resize_image(image, list(self.config.IMAGE_SHAPE[:2]))
        preprocessed_image = self.config.PREPROCESSING(tf.cast(resized_image, tf.float32))
        origin_image_shape = tf.shape(image)
        return {'input_images': preprocessed_image,
                'input_window': window,
                'origin_image_shapes':origin_image_shape,
                'image_ids':img_id}

    @tf.function
    def preprocessing_train(self, path, ann_ids):
        image = self.load_image(path)
        boxes, dataloader_class_ids =\
            tf.py_function(self.load_gt, (ann_ids,tf.shape(image)[0],tf.shape(image)[1]),(tf.float32, tf.int64))
    
        if self.augmentations is not None:
            image, boxes = \
                self.augment(image, boxes)

        resized_image, resized_boxes = \
            self.resize(image, boxes)
        
        preprocessed_image = self.config.PREPROCESSING(tf.cast(resized_image, tf.float32))

        rpn_match, rpn_bbox = \
            self.build_rpn_targets(dataloader_class_ids, resized_boxes)

        
        pooled_box = tf.zeros([self.config.MAX_GT_INSTANCES,4],dtype=tf.float32)
        pooled_class_id = tf.zeros([self.config.MAX_GT_INSTANCES],dtype=tf.int64)

        instance_count = tf.shape(boxes)[0]
        if instance_count>self.config.MAX_GT_INSTANCES:
            indices = tf.random.shuffle(tf.range(instance_count))[:self.config.MAX_GT_INSTANCES]
            indices = tf.expand_dims(indices,1)
        else:
            indices = tf.range(instance_count)
            indices = tf.expand_dims(indices,1)

        resized_boxes = tf.tensor_scatter_nd_update(pooled_box, indices, tf.gather(resized_boxes, tf.squeeze(indices,-1)))
        dataloader_class_ids = tf.tensor_scatter_nd_update(pooled_class_id, indices, tf.gather(dataloader_class_ids, tf.squeeze(indices, -1)))

        return {'input_images': preprocessed_image,
                'input_gt_boxes': resized_boxes,
                'dataloader_class_ids':dataloader_class_ids,
                'rpn_match':rpn_match,
                'rpn_bbox':rpn_bbox}


    def load_gt(self, ann_ids, h, w):
        ann_ids = ann_ids.numpy()
        ann_ids = [ann_id for ann_id in ann_ids if ann_id!=0]
        anns = self.dataset.coco.loadAnns(ann_ids)

        gts = [gt for gt in [self.load_ann(ann, (h,w)) for ann in anns] if gt is not None]
        if gts:
            boxes, dataloader_class_ids = list(zip(*gts))
            
            boxes = tf.cast(tf.stack(boxes), tf.float32)
            dataloader_class_ids = tf.cast(tf.stack(dataloader_class_ids), tf.int64)
        else:
            boxes = tf.zeros([0,4], dtype=tf.float32)
            dataloader_class_ids = tf.zeros([0], dtype=tf.int64)
        return boxes, dataloader_class_ids


    @tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.int64)])
    def padding_ann_ids(self, ann_ids):
        def f(ann_ids):
            padded_ann_ids = tf.zeros([self.config.MAX_GT_INSTANCES], dtype=tf.int64)
            indices = tf.expand_dims(tf.range(tf.shape(ann_ids)[0]),1)
            ann_ids = tf.tensor_scatter_nd_update(padded_ann_ids, indices, tf.gather(ann_ids, tf.range(tf.shape(ann_ids)[0])))
            return ann_ids
        
        ann_ids = tf.cast(ann_ids, tf.int64)

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
    def resize(self, image, bbox):
        origin_shape = tf.shape(image)
        resized_image, window = self.resize_image(image, list(self.config.IMAGE_SHAPE[:2]))
        resized_bbox = self.resize_box(bbox, origin_shape, window)
        return resized_image, resized_bbox
        

    @staticmethod
    @tf.function
    def resize_image(image, shape):
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
                                shape[:2],
                                preserve_aspect_ratio=True, 
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        shape = tf.cast(shape, tf.int32)
        # Get new height and width
        h = tf.shape(image)[0]
        w = tf.shape(image)[1]
        top_pad = (shape[0] - h) // 2
        bottom_pad = shape[0] - h - top_pad
        left_pad = (shape[1] - w) // 2
        right_pad = shape[1] - w - left_pad
        padding = tf.stack([(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)])
        image = tf.pad(image, padding, mode='constant', constant_values=0)
        window = tf.stack((top_pad, left_pad, h + top_pad, w + left_pad))
        return image, window
    

    def load_ann(self, ann, image_shape):
        dataloader_class_id = self.dataset.get_dataloader_class_id(ann['category_id'])

        x1,y1,w,h = ann['bbox']
        box = np.array((y1,x1,y1+h,x1+w))

        y1, x1, y2, x2 = np.round(box)
        area = (y2-y1)*(x2-x1)
        if area == 0:
            print(f'area of {ann["bbox"]} is 0')
            return None

        h = image_shape[0]
        w = image_shape[1]

        if ann['iscrowd']:
            # Use negative class ID for crowds
            dataloader_class_id *= -1
            # For crowd masks, annToMask() sometimes returns a mask
            # smaller than the given dimensions. If so, resize it.

        return box, dataloader_class_id


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
    def augment(self, image, bbox):
        def _augment(image,bbox):
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
            shape = image.shape[:2]

            bbox = bbox[:,[1,0,3,2]]
            bbi = BoundingBoxesOnImage.from_xyxy_array(bbox, shape)

            image, bbox = self.augmentations(image=image, bounding_boxes=bbi)
            bbox = bbox.to_xyxy_array()[:,[1,0,3,2]]

            return image,bbox
        
        return tf.py_function(_augment, 
                            [image,bbox],
                            (tf.uint8,tf.float32), name='augment')


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
        anchors = tf.cast(self.anchors, tf.float32)

        if tf.shape(gt_boxes)[0] == 0:
            rpn_bbox = tf.zeros([self.config.RPN_TRAIN_ANCHORS_PER_IMAGE,4], dtype=tf.float32)
            return rpn_match, rpn_bbox

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
            # crowd_overlaps = compute_overlaps(self.anchors, crowd_boxes)
            crowd_boxes = tf.ensure_shape(crowd_boxes,[None,4])
            crowd_overlaps = iou(anchors, crowd_boxes)
            crowd_iou_max = tf.cast(tf.reduce_max(crowd_overlaps, axis=1),tf.float32)
            no_crowd_bool = (crowd_iou_max < 0.001)
        else:
            # All anchors don't intersect a crowd
            no_crowd_bool = tf.ones([anchors.shape[0]], dtype=tf.bool)
        
        # Compute overlaps [num_anchors, num_gt_boxes]
        # overlaps = compute_overlaps(anchors, gt_boxes)
        gt_boxes = tf.ensure_shape(gt_boxes,[None,4])
        overlaps = iou(anchors, gt_boxes)

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
        anchor_iou_max = tf.ensure_shape(anchor_iou_max, [anchors.shape[0]])
        no_crowd_bool = tf.ensure_shape(no_crowd_bool, [anchors.shape[0]])
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
        # TODO: use box_refinement() rather than duplicating the code here
        gathered_anchores = tf.cast(tf.gather(anchors,ids), tf.float32)

        for i in tf.range(tf.shape(ids)[0]):
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
            rpn_bbox = rpn_bbox.write(i, tf.stack([(gt_center_y - a_center_y) / a_h, 
                                                        (gt_center_x - a_center_x) / a_w,
                                                        tf.math.log(gt_h / a_h), 
                                                        tf.math.log(gt_w / a_w),
                                                        ])/self.config.RPN_BBOX_STD_DEV)
        rpn_bbox = rpn_bbox.stack()

        return rpn_match, rpn_bbox