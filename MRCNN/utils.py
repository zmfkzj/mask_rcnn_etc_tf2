from copy import deepcopy
import math
import random
import shutil
import urllib.request
import warnings

import keras.api._v2.keras as keras
import numpy as np
import scipy
import skimage.transform
import tensorflow as tf

from MRCNN.config import Config


@tf.function
def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]. (y2, x2) is
    assumed to be outside the box.
    """
    box = tf.cast(box,tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.math.log(gt_height / height)
    dw = tf.math.log(gt_width / width)

    return tf.stack([dy, dx, dh, dw], axis=1)


############################################################
#  Anchors
############################################################

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)


@tf.function
def norm_boxes(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in normalized coordinates
    """
    h = shape[0]
    w = shape[1]
    scale = tf.stack([h - 1, w - 1, h - 1, w - 1])
    shift = tf.stack([0, 0, 1, 1])
    return tf.divide((tf.cast(boxes,tf.float32) - tf.cast(shift, tf.float32)), tf.cast(scale,tf.float32))



@tf.function
def denorm_boxes(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [N, (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in pixel coordinates
    """
    h = shape[0]
    w = shape[1]
    scale = tf.cast(tf.stack([h - 1, w - 1, h - 1, w - 1]), tf.float32)
    shift = tf.stack([0., 0., 1., 1.])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift),tf.int64)


def compute_backbone_shapes(config:Config):
    """Computes the width and height of each stage of the backbone network.

    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    backbone = deepcopy(config.BACKBONE)(input_shape=list(config.IMAGE_SHAPE), include_top=False)

    output_shapes = list(sorted(set([tuple(layer.output_shape[1:3]) for layer in backbone.layers]), reverse=True))
    output_shapes = [shape for shape in output_shapes if len(shape)==2 and np.all(config.IMAGE_SHAPE[:2]%np.array(shape)==0) and np.all(np.array(shape)!=1)]
    return np.array(output_shapes[-4:])


def unmold_detections(detections, original_image_shape, image_shape, window, mrcnn_mask=None):
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
    zero_ix = tf.where(detections[:, 4] == 0)[...,0]
    N = tf.cond(tf.shape(zero_ix)[0] > 0,
                lambda: zero_ix[0],
                lambda: tf.shape(detections)[0])

    # Extract boxes, class_ids, scores, and class-specific masks
    boxes = detections[:N, :4]
    class_ids = tf.cast(detections[:N, 4],tf.int64)
    scores = detections[:N, 5]
    if mrcnn_mask is not None:
        masks = mrcnn_mask[:N, :, :, class_ids]

    # Translate normalized coordinates in the resized image to pixel
    # coordinates in the original image before resizing
    window = norm_boxes(window, image_shape[:2])
    wy1 = window[0]
    wx1 = window[1]
    wy2 = window[2]
    wx2 = window[3]
    shift = tf.stack([wy1, wx1, wy1, wx1])
    wh = wy2 - wy1  # window height
    ww = wx2 - wx1  # window width
    scale = tf.stack([wh, ww, wh, ww])
    # Convert boxes to normalized coordinates on the window
    boxes = tf.divide(boxes - shift, scale)
    # Convert boxes to pixel coordinates on the original image
    boxes = denorm_boxes(boxes, original_image_shape[:2])

    # Filter out detections with zero area. Happens in early training when
    # network weights are still random
    include_ix = tf.where( (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) > 0)[...,0]
    exclude_ix = tf.where( (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[...,0]
    boxes = tf.cond(tf.shape(exclude_ix)[0] > 0, 
                    lambda: tf.gather(boxes, include_ix, axis=0), 
                    lambda: boxes)
    class_ids = tf.cond(tf.shape(exclude_ix)[0] > 0, 
                        lambda: tf.gather(class_ids, include_ix, axis=0), 
                        lambda: class_ids)
    scores = tf.cond(tf.shape(exclude_ix)[0] > 0, 
                    lambda: tf.gather(scores, include_ix, axis=0), 
                        lambda: scores)
    if mrcnn_mask is not None:
        masks = tf.cond(tf.shape(exclude_ix)[0] > 0, 
                        lambda: tf.gather(masks, include_ix, axis=0), 
                        lambda: masks)
    N = tf.cond(tf.shape(exclude_ix)[0] > 0, 
                lambda: tf.shape(class_ids)[0], 
                lambda: N)

    if mrcnn_mask is not None:
        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in tf.range(N):
            # Convert neural network mask to full size mask
            full_mask = unmold_mask(masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
        full_masks = tf.cond(N>0,
                            lambda: tf.stack(full_masks, axis=-1),
                            lambda: tf.zeros(original_image_shape[:2] + (0,)))
    else:
        full_masks = None

    return boxes, class_ids, scores, full_masks

@tf.function
def unmold_mask(mask, bbox, image_shape):
    """Converts a mask generated by the neural network to a format similar
    to its original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.

    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.5
    y1 = bbox[0]
    x1 = bbox[1]
    y2 = bbox[2]
    x2 = bbox[3]
    mask = tf.image.resize(mask, (y2 - y1, x2 - x1), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask = tf.cast(tf.where(mask >= threshold, 1, 0),tf.bool)

    # Put the mask in the right location.
    full_mask = tf.zeros(image_shape[:2], dtype=tf.bool)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask


@tf.function
def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    box = tf.cast(box, tf.float32)
    boxes = tf.cast(boxes, tf.float32)
    box_area = tf.cast(box_area, tf.float32)
    boxes_area = tf.cast(boxes_area, tf.float32)
    # Calculate intersection areas
    y1 = tf.maximum(box[0], boxes[:, 0])
    y2 = tf.minimum(box[2], boxes[:, 2])
    x1 = tf.maximum(box[1], boxes[:, 1])
    x2 = tf.minimum(box[3], boxes[:, 3])
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


@tf.function
def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = tf.vectorized_map(lambda args: compute_iou(args[0], boxes2, args[1], area2),
                                 [boxes1, area1])

    return overlaps
