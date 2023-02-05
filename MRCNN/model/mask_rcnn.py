import re

import keras.api._v2.keras as keras
import keras.api._v2.keras.layers as KL
import keras.api._v2.keras.models as KM
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_models as tfm
from official.vision.ops.iou_similarity import iou
from pycocotools.coco import COCO

from MRCNN import utils
from MRCNN.config import Config
from MRCNN.layer.roialign import parse_image_meta_graph
from MRCNN.loss import (mrcnn_bbox_loss_graph, mrcnn_class_loss_graph,
                        mrcnn_mask_loss_graph, rpn_bbox_loss_graph,
                        rpn_class_loss_graph)

from ..layer import (DetectionLayer, DetectionTargetLayer, ProposalLayer,
                     PyramidROIAlign)
from ..model_utils.miscellenous_graph import norm_boxes_graph
from ..utils import compute_backbone_shapes, log
from . import RPN, FPN_classifier, FPN_mask, Neck


class MaskRCNN(KM.Model):
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, config:Config, coco_json):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        super().__init__(name='mask_rcnn')
        self.config = config
        self.coco = COCO(coco_json)

        if isinstance(config.GPUS, int):
            gpus = [config.GPUS]
        else:
            gpus = config.GPUS
        self.strategy = tf.distribute.MirroredStrategy(devices=[f'/gpu:{gpu_id}' for gpu_id in gpus], cross_device_ops=config.CROSS_DEVICE_OPS)

        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")


        #parts
        self.meta_conv1 = KL.Conv2D(64,(7,7),strides=(2,2),name='meta_conv1', use_bias=True)
        self.meta_cls_score = KL.Dense(config.NUM_CLASSES, kernel_initializer=tf.initializers.HeUniform())

        self.backbone = self.make_backbone_model(config.BACKBONE, config.PREPROCESSING)
        self.neck = Neck(config)

        self.rpn = RPN(self.config.RPN_ANCHOR_STRIDE, len(self.config.RPN_ANCHOR_SCALES)*len(self.config.RPN_ANCHOR_RATIOS), name='rpn_model')

        self.ROIAlign_classifier = PyramidROIAlign([config.POOL_SIZE, config.POOL_SIZE], name="roi_align_classifier")
        self.ROIAlign_mask = PyramidROIAlign([config.MASK_POOL_SIZE, config.MASK_POOL_SIZE], name="roi_align_mask")

        self.fpn_classifier = FPN_classifier(self.config.POOL_SIZE, self.config.NUM_CLASSES, fc_layers_size=self.config.FPN_CLASSIF_FC_LAYERS_SIZE)
        self.fpn_mask = FPN_mask(self.config.NUM_CLASSES)

        self.concats = [KL.Concatenate(axis=1, name=n) for n in ["rpn_class_logits", "rpn_class", "rpn_bbox"]]
        self.anchors = self.get_anchors(self.config.IMAGE_SHAPE)

        #excuting models
        self.predict_model = self.make_predict_model()
        self.test_model = self.make_test_model()
        self.train_model = self.make_train_model()
    

    def predict_step(self, data):
        input_images, input_window, origin_image_shapes = data
        detections, mrcnn_mask = self.predict_model(input_images, input_window)

        all_rois = []
        all_class_ids = []
        all_scores = []
        all_masks = []

        batch_size = tf.shape(input_images)[0]
        for i in tf.range(batch_size):
            boxes, class_ids, scores, full_masks = \
                            self.unmold_detections(detections[i],mrcnn_mask[i],origin_image_shapes[i],tf.shape(input_images[i]), input_window[i])
            all_rois.append(boxes)
            all_class_ids.append(class_ids)
            all_scores.append(scores)
            all_masks.append(full_masks)
        
        all_rois = tf.stack(all_rois)
        all_class_ids = tf.stack(all_class_ids)
        all_scores = tf.stack(all_scores)
        all_masks = tf.stack(all_masks)

        return all_rois, all_class_ids, all_scores, all_masks
            


    def test_step(self, data):
        detections, mrcnn_mask = self.test_model(*data)


    def train_step(self, data):
        return super().train_step(data)   

    def make_predict_model(self):
        input_image = KL.Input(self.config.IMAGE_SHAPE, dtype=tf.uint8, name='input_image')
        input_window = KL.Input(shape=(), name="input_window")

        backbone_output = self.backbone(input_image)
        P2,P3,P4,P5,P6 = self.neck(*backbone_output)
        

        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(self.rpn(p))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        outputs = list(zip(*layer_outputs))
        outputs = [c(list(o)) for o, c in zip(outputs, self.concats)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        batch_size = tf.shape(input_image)[0]
        anchors = tf.broadcast_to(self.anchors, tf.concat([(batch_size,),tf.shape(self.anchors)],-1))
        proposal_count = self.config.POST_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(nms_threshold=self.config.RPN_NMS_THRESHOLD, name="ROI", config=self.config)([rpn_class, rpn_bbox, anchors, proposal_count])
        
        roi_cls_feature = self.ROIAlign_classifier(rpn_rois, self.config.IMAGE_SHAPE, mrcnn_feature_maps)

        # Network Heads
        # Proposal classifier and BBox regressor heads
        mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.fpn_classifier(roi_cls_feature, training=False)
        detections = DetectionLayer(self.config, name="mrcnn_detection")(rpn_rois, mrcnn_class, mrcnn_bbox, self.config.IMAGE_SHAPE, input_window)

        # Create masks for detections
        detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
        roi_seg_feature = self.ROIAlign_mask(detection_boxes, self.config.IMAGE_SHAPE, mrcnn_feature_maps)
        mrcnn_mask = self.fpn_mask(roi_seg_feature, train_bn=False)

        model = keras.Model([input_image, input_window],
                            [detections, mrcnn_mask],
                            name='mask_rcnn')
        return model
    

    def make_test_model(self):
        return self.make_predict_model()
    

    def make_train_model(self):
        input_image = KL.Input( shape=[None, None, self.config.IMAGE_SHAPE[2]], name="input_image")
        active_class_ids = KL.Input(shape=[self.config.NUM_CLASSES], name="input_class_ids")
        # RPN GT
        input_rpn_match = KL.Input( shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
        input_rpn_bbox = KL.Input( shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

        # Detection GT (class IDs, bounding boxes, and masks)
        # 1. GT Class IDs (zero padded)
        input_gt_class_ids = KL.Input( shape=[None], name="input_gt_class_ids", dtype=tf.int32)
        # 2. GT Boxes in pixels (zero padded)
        # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
        input_gt_boxes = KL.Input( shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
        # Normalize coordinates
        gt_boxes = KL.Lambda(lambda x: norm_boxes_graph( x, tf.shape(input_image)[1:3]))(input_gt_boxes)
        # 3. GT Masks (zero padded)
        # [batch, height, width, MAX_GT_INSTANCES]
        input_gt_masks = KL.Input( shape=[self.config.MINI_MASK_SHAPE[0], self.config.MINI_MASK_SHAPE[1], None], name="input_gt_masks", dtype=bool)

        backbone_output = self.backbone(input_image)
        P2,P3,P4,P5,P6 = self.neck(*backbone_output)
        

        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(self.rpn(p))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        outputs = list(zip(*layer_outputs))
        outputs = [c(list(o)) for o, c in zip(outputs, self.concats)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        batch_size = tf.shape(input_image)[0]
        anchors = tf.broadcast_to(self.anchors, tf.concat([(batch_size,),tf.shape(self.anchors)],-1))
        proposal_count = self.config.POST_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(nms_threshold=self.config.RPN_NMS_THRESHOLD, name="ROI", config=self.config)([rpn_class, rpn_bbox, anchors, proposal_count])


        # Generate detection targets
        # Subsamples proposals and generates target outputs for training
        # Note that proposal class IDs, gt_boxes, and gt_masks are zero
        # padded. Equally, returned rois and targets are zero padded.
        rois, target_class_ids, target_bbox, target_mask =\
            DetectionTargetLayer(self.config, name="proposal_targets")([rpn_rois, input_gt_class_ids, gt_boxes, input_gt_masks])
        
        roi_cls_features = self.ROIAlign_classifier(rois, self.config.IMAGE_SHAPE, mrcnn_feature_maps)
        roi_mask_features = self.ROIAlign_mask(rois, self.config.IMAGE_SHAPE, mrcnn_feature_maps)

        # Network Heads
        # TODO: verify that this handles zero padded ROIs
        mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.fpn_classifier(roi_cls_features)
        mrcnn_mask = self.fpn_mask(roi_mask_features)

        # Losses
        rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")( [input_rpn_match, rpn_class_logits])
        rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(batch_size, *x), name="rpn_bbox_loss")( [input_rpn_bbox, input_rpn_match, rpn_bbox])
        class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")( [target_class_ids, mrcnn_class_logits, active_class_ids])
        bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")( [target_bbox, target_class_ids, mrcnn_bbox])
        mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")( [target_mask, target_class_ids, mrcnn_mask])

        # Model
        inputs = [input_image, active_class_ids, input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]
        outputs = [rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]

        model = keras.Model(inputs, outputs, name='mask_rcnn')

        reg_losses = [keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
                        for w in model.trainable_weights if 'gamma' not in w.name and 'beta' not in w.name]
        model.add_loss(tf.add_n(reg_losses))
        model.add_loss(outputs)
        return model



    def make_backbone_model(self, backbone, preprocessing_function):
        backbone:KM.Model = backbone(include_top=False)

        i = KL.Input(self.config.IMAGE_SHAPE, dtype=tf.uint8)
        x = tf.cast(i, tf.float32)
        x = preprocessing_function(x)
        x = backbone(x)

        output = []
        for i,layer in enumerate(backbone.layers[:-1]):
            if layer.output_shape[1:3] != backbone.layers[i+1].output_shape[1:3]:
                output.append(layer.output)
        output.append(x)

        model = keras.Model(inputs=[i], outputs=output[-4:])
        
        return model
        

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model:KM.Model = keras_model or self

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if isinstance(layer, KM.Model):
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainable layer names
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        key = tuple(image_shape)
        if not key in self._anchor_cache:
            # Generate Anchors
            a = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # TODO: Remove this after the notebook are refactored to not use it
            self.anchors = a
            # Normalize coordinates
            self._anchor_cache[key] = utils.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[key]
    
    def load_weights(self, filepath):
        import h5py

        f = h5py.File(filepath, mode='r')
        saved_root_layer_names = [name if isinstance(name, str) else name.decode('utf-8') for name in f.attrs['layer_names']]
        saved_weight_names = []
        saved_weight_values = []
        for ln in saved_root_layer_names:
            for wn in f[ln].attrs['weight_names']:
                if not isinstance(wn, str):
                    wn = wn.decode('utf-8') 
                saved_weight_values.append(f[f'/{ln}/{wn}'])
                saved_weight_names.append('/'.join(wn.split('/')[-2:]))
        saved_weights = dict(zip(saved_weight_names,saved_weight_values))
        model_layers = self.collect_layers(self)
        for l in model_layers.values():
            for w in l.weights:
                weight_name = '/'.join(w.name.split('/')[-2:])
                if (weight_name in saved_weights) and (tuple(w.shape)==saved_weights[weight_name].shape):
                    w.assign(np.array(saved_weights[weight_name]))
                else:
                    print(f'{weight_name}\ can\'t assign')
        return self

    def collect_layers(self, model):
        layers = {}
        if isinstance(model, KM.Model):
            for layer in model.layers:
                layers.update(self.collect_layers(layer))
        else:
            layers[model.name]=model
        return layers

    @tf.function
    def unmold_detections(self, detections, mrcnn_mask, original_image_shape, image_shape, window):
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
        zero_ix = tf.where(detections[:, 4] == 0)[0]
        N = tf.cond(tf.shape(zero_ix)[0],
                    lambda: zero_ix[0],
                    lambda: tf.shape(detections)[0])

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = tf.cast(detections[:N, 4],tf.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[:N, :, :, class_ids]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = utils.norm_boxes(window, image_shape[:2])
        wy1 = window[0]
        wx1 = window[1]
        wy2 = window[2]
        wx2 = window[3]
        shift = tf.sta([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = tf.sta([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = tf.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = tf.where( (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        boxes = tf.cond(tf.shape(exclude_ix)[0] > 0, lambda: np.delete(boxes, exclude_ix, axis=0), lambda: boxes)
        class_ids = tf.cond(tf.shape(exclude_ix)[0] > 0, lambda: np.delete(class_ids, exclude_ix, axis=0), lambda: class_ids)
        scores = tf.cond(tf.shape(exclude_ix)[0] > 0, lambda: np.delete(scores, exclude_ix, axis=0), lambda: scores)
        masks = tf.cond(tf.shape(exclude_ix)[0] > 0, lambda: np.delete(masks, exclude_ix, axis=0), lambda: masks)
        N = tf.cond(tf.shape(exclude_ix)[0] > 0, lambda: tf.shape(class_ids)[0], lambda: N)

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in tf.range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
        full_masks = tf.cond(N>0,
                            lambda: tf.stack(full_masks, axis=-1),
                            lambda: tf.zeros(original_image_shape[:2] + (0,)))

        return boxes, class_ids, scores, full_masks