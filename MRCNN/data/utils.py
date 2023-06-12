from functools import partial

import albumentations as A
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_models as tfm
from pycocotools import mask as maskUtils

from MRCNN.config import Config
from MRCNN.data.dataset import Dataset
from MRCNN.utils import compute_backbone_shapes, generate_pyramid_anchors


def load_gt(dataset:Dataset, ann_ids, h, w):
    ann_ids = ann_ids.numpy()
    ann_ids = [ann_id for ann_id in ann_ids if ann_id!=0]
    anns = dataset.coco.loadAnns(ann_ids)

    gts = [gt for gt in [load_ann(dataset, ann, (h,w)) for ann in anns] if gt is not None]
    if gts:
        boxes, loader_class_ids, masks = list(zip(*gts))
        
        boxes = tf.cast(tf.stack(boxes), tf.float16)
        loader_class_ids = tf.cast(tf.stack(loader_class_ids), tf.int16)
        masks = tf.cast(tf.stack(masks), tf.bool)
    else:
        boxes = tf.zeros([0,4], dtype=tf.float16)
        loader_class_ids = tf.zeros([0], dtype=tf.int16)
        masks = tf.zeros([0,h,w], dtype=tf.bool)
    return boxes, loader_class_ids, masks


@tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.int64),
                              tf.TensorSpec(shape=(), dtype=tf.int16)])
def padding_ann_ids(ann_ids, max_gt_instances):
    def f(ann_ids):
        padded_ann_ids = tf.zeros([max_gt_instances], dtype=tf.int64)
        indices = tf.expand_dims(tf.range(tf.shape(ann_ids)[0]),1)
        ann_ids = tf.tensor_scatter_nd_update(padded_ann_ids, indices, tf.gather(ann_ids, tf.range(tf.shape(ann_ids)[0])))
        return ann_ids
    
    ann_ids = tf.cast(ann_ids, tf.int64)

    if tf.cast(tf.shape(ann_ids)[0], tf.int16)>max_gt_instances:
        ann_ids = tf.random.shuffle(ann_ids)[:max_gt_instances]
        ann_ids = f(ann_ids)
    else:
        ann_ids = tf.stack(ann_ids)
        ann_ids = f(ann_ids)
    
    return ann_ids


@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)])
def load_image(path):
    img_raw = tf.io.read_file(path)
    image = tf.io.decode_image(img_raw,3)
    return image


@tf.function
def normalize(image, pixel_mean, pixel_std):
    image = tf.cast(image, tf.float16)
    return (image - tf.cast(pixel_mean,tf.float16)) / tf.cast(pixel_std,tf.float16)


@tf.function
def resize(image, bbox, mask, resize_shape, mini_mask_shape):
    origin_shape = tf.shape(image)
    resized_image, window = resize_image(image, resize_shape)
    resized_bbox = resize_box(bbox, origin_shape, window)
    masks = minimize_mask(bbox, mask, mini_mask_shape)
    return resized_image, resized_bbox, masks


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
                            method=tf.image.ResizeMethod.BILINEAR)

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


def load_ann(dataset: Dataset, ann, image_shape):
    dataloader_class_id = dataset.get_loader_class_id(ann['category_id'])

    x1,y1,w,h = ann['bbox']
    box = np.array((y1,x1,y1+h,x1+w))

    y1, x1, y2, x2 = np.round(box)
    area = (y2-y1)*(x2-x1)
    if area == 0:
        return None

    h = image_shape[0]
    w = image_shape[1]

    if ann['iscrowd']:
        # Use negative class ID for crowds
        dataloader_class_id *= -1
        # For crowd masks, annToMask() sometimes returns a mask
        # smaller than the given dimensions. If so, resize it.

    mask = annToMask(ann, h, w)
    return box, dataloader_class_id, mask


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
            m = tf.zeros(mini_shape, tf.bool)

        # Resize with bilinear interpolation
        m = tf.expand_dims(m,2)
        m = tf.cast(m, tf.uint8)
        m = tf.image.resize(m, mini_shape, method=tf.image.ResizeMethod.BILINEAR)
        m = tf.squeeze(m, -1)
        m = tf.cast(m, tf.bool)
        return m
    
    mini_mask = tf.map_fn(f, [mask,bbox],fn_output_signature=tf.TensorSpec(shape=mini_shape, dtype=tf.bool), )
    return mini_mask


def annToRLE(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = ann['segmentation']
    if isinstance(segm, list):
        if not segm:
            segm = [[0.,0.,0.,0.,0.,0.]]

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


def annToMask(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = annToRLE(ann, height, width)
    m = maskUtils.decode(rle)
    return m


@tf.function
def resize_box(bboxes, origin_shape, window):
    origin_shape = tf.cast(origin_shape, tf.float16)
    window = tf.cast(window, tf.float16)
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
def padding_bbox(boxes, cls_ids, max_gt_instances):
    boxes = tf.pad(boxes,[[0,max_gt_instances-tf.shape(boxes)[0]],[0,0]])
    cls_ids = tf.reshape(cls_ids, [-1,1])
    cls_ids = tf.pad(cls_ids,[[0,max_gt_instances-tf.shape(cls_ids)[0]],[0,0]])
    cls_ids = tf.squeeze(cls_ids, 1)
    return boxes, cls_ids


@tf.function
def build_rpn_targets(gt_class_ids, gt_boxes, anchors,rpn_train_anchors_per_image,rpn_bbox_std_dev):
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
    rpn_match = tf.zeros([anchors.shape[0]], dtype=tf.int32)
    rpn_bbox = tf.zeros([rpn_train_anchors_per_image,4], dtype=tf.float16)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    anchors = tf.cast(anchors, tf.float16)

    if tf.shape(gt_boxes)[0] == 0:
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
        crowd_overlaps = tfm.vision.iou_similarity.iou(tf.cast(anchors, tf.float32), tf.cast(crowd_boxes, tf.float32))
        crowd_overlaps = tf.cast(crowd_overlaps, tf.float16)
        crowd_iou_max = tf.cast(tf.reduce_max(crowd_overlaps, axis=1),tf.float16)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        # All anchors don't intersect a crowd
        no_crowd_bool = tf.ones([anchors.shape[0]], dtype=tf.bool)
    
    # Compute overlaps [num_anchors, num_gt_boxes]
    # overlaps = compute_overlaps(anchors, gt_boxes)
    gt_boxes = tf.ensure_shape(gt_boxes,[None,4])
    overlaps = tfm.vision.iou_similarity.iou(tf.cast(anchors, tf.float32), tf.cast(gt_boxes, tf.float32))
    overlaps = tf.cast(overlaps, tf.float16)

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
    anchor_iou_argmax = tf.cast(tf.argmax(overlaps, axis=1), tf.int32)
    
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
    extra = tf.shape(ids)[0] - (rpn_train_anchors_per_image // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = tf.random.shuffle(ids)[:extra]
        rpn_match = tf.tensor_scatter_nd_update(rpn_match, 
                                                tf.expand_dims(ids,1),
                                                tf.zeros([tf.shape(ids)[0]],tf.int32))

    # Same for negative proposals
    ids = tf.transpose(tf.where(rpn_match == -1))[0]
    extra = tf.shape(ids)[0] - (rpn_train_anchors_per_image - tf.reduce_sum(tf.cast(rpn_match == 1,tf.int32)))
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
    gathered_anchores = tf.cast(tf.gather(anchors,ids), tf.float16)

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
        _rpn_bbox = tf.stack([(gt_center_y - a_center_y) / a_h, 
                              (gt_center_x - a_center_x) / a_w, 
                              tf.math.log(gt_h / a_h), 
                              tf.math.log(gt_w / a_w)]) \
                                /rpn_bbox_std_dev
        
        rpn_bbox = tf.tensor_scatter_nd_update(rpn_bbox, [[i]], [_rpn_bbox])
    
    return rpn_match, rpn_bbox


def get_anchors(config:Config):
    backbone_shapes = compute_backbone_shapes(config)
    anchors = generate_pyramid_anchors(
                        config.RPN_ANCHOR_SCALES,
                        config.RPN_ANCHOR_RATIOS,
                        backbone_shapes,
                        config.BACKBONE_STRIDES,
                        config.RPN_ANCHOR_STRIDE)
    return anchors


def preprocessing_predict(path, resize_shape, pixel_mean, pixel_std):
    image = load_image(path)
    resized_image, window = resize_image(image, resize_shape)
    norm_image = normalize(resized_image, pixel_mean, pixel_std)
    origin_image_shape = tf.shape(image)
    return norm_image, window, origin_image_shape, path


def preprocessing_test(path,resize_shape, pixel_mean, pixel_std, img_id):
    image = load_image(path)
    resized_image, window = resize_image(image, resize_shape)
    norm_image = normalize(resized_image, pixel_mean, pixel_std)
    origin_image_shape = tf.shape(image)
    return norm_image, window, origin_image_shape, img_id


def preprocessing_train(path, 
                        resize_shape, 
                        pixel_mean, 
                        pixel_std, 
                        ann_ids, 
                        anchors, 
                        rpn_train_anchors_per_image,
                        rpn_bbox_std_dev,
                        max_gt_instances, 
                        mini_mask_shape,
                        dataset,
                        augmentor=None):
    image = load_image(path)
    boxes, loader_class_ids, masks =\
        tf.py_function(lambda ann_ids, h, w: load_gt(dataset, ann_ids, h, w), 
                       (ann_ids,tf.shape(image)[0],tf.shape(image)[1]),(tf.float16, tf.int16, tf.bool))

    if augmentor is not None:
        image, boxes, masks, loader_class_ids =  tf.py_function(augmentor, (image, boxes, masks, loader_class_ids), (tf.uint8, tf.float16, tf.bool, tf.int16))

    resized_image, resized_boxes, minimized_masks = resize(image, boxes, masks, resize_shape, mini_mask_shape)
    
    norm_image = normalize(resized_image, pixel_mean, pixel_std)

    rpn_match, rpn_bbox = build_rpn_targets(loader_class_ids, resized_boxes, anchors,rpn_train_anchors_per_image,rpn_bbox_std_dev)

    pooled_box = tf.zeros([max_gt_instances,4],dtype=tf.float16)
    pooled_class_id = tf.zeros([max_gt_instances],dtype=tf.int16)
    pooled_mask = tf.zeros([max_gt_instances,*mini_mask_shape],dtype=tf.bool)

    instance_count = tf.shape(boxes)[0]
    if instance_count>max_gt_instances:
        indices = tf.random.shuffle(tf.range(instance_count))[:max_gt_instances]
        indices = tf.expand_dims(indices,1)
    else:
        indices = tf.range(instance_count)
        indices = tf.expand_dims(indices,1)

    resized_boxes = tf.tensor_scatter_nd_update(pooled_box, indices, tf.gather(resized_boxes, tf.squeeze(indices,-1)))
    loader_class_ids = tf.tensor_scatter_nd_update(pooled_class_id, indices, tf.gather(loader_class_ids, tf.squeeze(indices, -1)))
    masks = tf.tensor_scatter_nd_update(pooled_mask, indices, tf.gather(minimized_masks, tf.squeeze(indices,-1)))

    masks = tf.transpose(masks, [1,2,0])

    return norm_image, resized_boxes, loader_class_ids, rpn_match, rpn_bbox, masks


def get_augmentor(config: Config):
    augment_list = config.AUGMENTORS

    transforms = []
    for a in augment_list:
        transforms.append(getattr(A, a)(p=0.5))

    augmentor = partial(_augmentor,
                         transforms=transforms)
    return augmentor
    
    
def _augmentor(img, box, masks, classes, transforms):
    img = img.numpy().round().astype(np.uint8)
    box = box.numpy()[:,[1,0,3,2]]
    masks = [m for m in masks.numpy().astype(np.uint8)]
    classes = classes.numpy()

    box_with_label = np.concatenate([box, np.expand_dims(classes, -1)], -1)


    transform = A.Compose(transforms,
                    bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.1))
    
    t = transform(image=img,masks=masks,bboxes=box_with_label)
    t_img = t['image']
    t_box_with_label = np.array(t['bboxes'])
    if t_box_with_label.size == 0:
        t_box_with_label = np.zeros([0,5])
    t_box = t_box_with_label[:,:4]
    t_classes = t_box_with_label[:,4]
    if t['masks']:
        t_masks = np.stack(t['masks']).astype(np.bool_)
    else:
        t_masks = np.zeros([0, *t_img.shape[:2]])
    
    t_box = t_box[:, [1,0,3,2]]

    return t_img, t_box, t_masks, t_classes

