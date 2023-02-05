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
def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)]. Note that (y2, x2) is outside the box.
    deltas: [N, (dy, dx, log(dh), log(dw))]
    """
    boxes = tf.cast(boxes,tf.float32)
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.math.exp(deltas[:, 2])
    width *= tf.math.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    return tf.stack([y1, x1, y2, x2], axis=1)


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


@tf.function
def resize_image(image, max_dim):
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
    image = tf.image.resize(image, (max_dim, max_dim), preserve_aspect_ratio=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # Get new height and width
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    top_pad = (max_dim - h) // 2
    bottom_pad = max_dim - h - top_pad
    left_pad = (max_dim - w) // 2
    right_pad = max_dim - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(image, padding, mode='constant', constant_values=0)
    window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, padding


@tf.function
def resize_mask(mask, resize_shape, padding):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    mask = tf.image.resize(mask, resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, preserve_aspect_ratio=True)
    mask = tf.pad(mask, padding, mode='constant', constant_values=0)
    return mask


@tf.function
def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to reduce memory load.
    Mini-masks can be resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    """
    mini_shape = tf.concat([mini_shape, [0]])
    mini_mask = tf.zeros(mini_shape, dtype=tf.bool)
    for i in tf.range(tf.shape(mask)[-1]):
        # Pick slice and cast to bool in case load_mask() returned wrong dtype
        m = tf.cast(mask[:, :, i],tf.bool)
        y1 = bbox[i][:4][0]
        x1 = bbox[i][:4][1]
        y2 = bbox[i][:4][2]
        x2 = bbox[i][:4][3]
        m = m[y1:y2, x1:x2]
        if tf.size(m) == 0:
            raise Exception("Invalid bounding box with area of zero")
        # Resize with bilinear interpolation
        m = tf.image.resize(m, mini_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        tf.tensor_scatter_nd_update()
        mini_mask = tf.concat([mini_mask, tf.cast(tf.round(m),tf.bool)], axis=-1)
    return mini_mask


@tf.function
def expand_mask(bbox, mini_mask, image_shape):
    """Resizes mini masks back to image size. Reverses the change
    of minimize_mask().

    See inspect_data.ipynb notebook for more details.
    """
    mask = np.zeros(image_shape[:2] + (mini_mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mini_mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        h = y2 - y1
        w = x2 - x1
        # Resize with bilinear interpolation
        m = tf.image.resize(m, (h, w), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mask[y1:y2, x1:x2, i] = np.around(m).astype(np.bool)
    return mask


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
    return tf.cast(tf.divide((boxes - shift), scale),tf.float32)



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
    scale = tf.stack([h - 1, w - 1, h - 1, w - 1])
    shift = tf.stack([0, 0, 1, 1])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift),tf.int32)


def compute_backbone_shapes(config:Config):
    """Computes the width and height of each stage of the backbone network.

    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    backbone = config.BACKBONE(input_shape=list(config.IMAGE_SHAPE), include_top=False)

    output_shapes = list(sorted(set([tuple(layer.output_shape[1:3]) for layer in backbone.layers]), reverse=True))
    output_shapes = [shape for shape in output_shapes if len(shape)==2 and np.all(config.IMAGE_SHAPE[:2]%np.array(shape)==0) and np.all(np.array(shape)!=1)]
    return np.array(output_shapes[-5:])