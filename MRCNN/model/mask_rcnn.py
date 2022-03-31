import re
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
import cv2

from MRCNN import utils
from MRCNN.config import Config
from MRCNN.loss import mrcnn_bbox_loss_graph, mrcnn_class_loss_graph, mrcnn_mask_loss_graph, rpn_bbox_loss_graph, rpn_class_loss_graph
from MRCNN.layer.roialign import parse_image_meta_graph
from . import FPN_classifier, FPN_mask, Backbone2FPN, RPN
from ..layer import DetectionLayer, DetectionTargetLayer, ProposalLayer, PyramidROIAlign
from ..model_utils.miscellenous_graph import norm_boxes_graph
from ..utils import log, compute_backbone_shapes


class MaskRCNN(KM.Model):
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, config:Config):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        super().__init__(name='mask_rcnn')
        self.config = config

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


        self.meta_conv1 = KL.Conv2D(64,(7,7),strides=(2,2),name='meta_conv1', use_bias=True)
        self.conv1 = KL.Conv2D(64,(7,7),strides=(2,2),name='conv1', use_bias=True)

        self.backbone2fpn = Backbone2FPN(config)
        self.rpn = RPN(self.config.RPN_ANCHOR_STRIDE, len(self.config.RPN_ANCHOR_RATIOS), name='rpn_model')

        self.ROIAlign_classifier = PyramidROIAlign([config.POOL_SIZE, config.POOL_SIZE], name="roi_align_classifier")
        self.ROIAlign_mask = PyramidROIAlign([config.MASK_POOL_SIZE, config.MASK_POOL_SIZE], name="roi_align_mask")

        self.fpn_classifier = FPN_classifier(self.config.POOL_SIZE, self.config.NUM_CLASSES, fc_layers_size=self.config.FPN_CLASSIF_FC_LAYERS_SIZE)
        self.fpn_mask = FPN_mask(self.config.NUM_CLASSES)

        self.concats = [KL.Concatenate(axis=1, name=n) for n in ["rpn_class_logits", "rpn_class", "rpn_bbox"]]

        self.anchors = self.get_anchors(self.config.IMAGE_SHAPE)

        self.attentions_layer = KL.Dense(config.TOP_DOWN_PYRAMID_SIZE, kernel_initializer=tf.initializers.HeUniform(), activation='sigmoid')
        self.meta_cls_score = KL.Dense(config.NUM_CLASSES, kernel_initializer=tf.initializers.HeUniform())

        with self.config.STRATEGY.scope():
            inputs = (tf.zeros([2,512,512,3]), tf.zeros([2,20]))
            attention = tf.zeros([5, config.TOP_DOWN_PYRAMID_SIZE])
            self.config.STRATEGY.run(self, args=(*inputs,), kwargs={'training':False, 'attentions':attention})


    def call(self, input_image, 
                    input_image_meta=None, 
                    input_rpn_match=None, 
                    input_rpn_bbox=None,
                    input_gt_class_ids=None, 
                    input_gt_boxes=None, 
                    input_gt_masks=None, 
                    input_rois = None,
                    attentions = None,
                    training=True):

        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        batch_size = input_image.shape[0]
        if batch_size == 0:
            rpn_count = tf.reduce_sum([tf.reduce_prod(tf.shape(f)[1:3])*len(self.config.RPN_ANCHOR_RATIOS) for f in rpn_feature_maps])
            rpn_bbox = tf.zeros([batch_size,rpn_count,4], dtype=tf.float32)
            
            if training:
                rpn_class_loss = tf.zeros((), dtype=tf.float32)
                rpn_bbox_loss = tf.zeros((), dtype=tf.float32)
                class_loss = tf.zeros((), dtype=tf.float32)
                bbox_loss = tf.zeros((), dtype=tf.float32)
                mask_loss = tf.zeros((), dtype=tf.float32)
                output = [rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
                return output
            else:
                detections = tf.zeros([batch_size,self.config.DETECTION_MAX_INSTANCES,6], dtype=tf.float32)
                mrcnn_class = tf.zeros([batch_size,self.config.POST_NMS_ROIS_INFERENCE,self.config.NUM_CLASSES], dtype=tf.float32)
                mrcnn_bbox = tf.zeros([batch_size,self.config.POST_NMS_ROIS_INFERENCE,self.config.NUM_CLASSES,4], dtype=tf.float32)
                rpn_rois = tf.zeros([batch_size,self.config.POST_NMS_ROIS_INFERENCE,4], dtype=tf.float32)
                rpn_class = tf.zeros([batch_size,rpn_count,2], dtype=tf.float32)
                mrcnn_mask = tf.zeros([batch_size,int(self.config.POST_NMS_ROIS_INFERENCE/10),self.config.MASK_SHAPE[0],self.config.MASK_SHAPE[1],self.config.NUM_CLASSES], dtype=tf.float32)
                output = [detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, rpn_rois, rpn_class, rpn_bbox]
                return output

        active_class_ids = KL.Lambda(lambda t: parse_image_meta_graph(t))(input_image_meta)["active_class_ids"]
        x = KL.ZeroPadding2D(padding=(3,3))(input_image)
        x = self.conv1(x)
        P2, P3, P4, P5, P6 = self.backbone2fpn(x)

        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        # Anchors
        anchors = tf.broadcast_to(self.anchors, tf.concat([(batch_size,),tf.shape(self.anchors)],-1))

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
        proposal_count = self.config.POST_NMS_ROIS_TRAINING if training else self.config.POST_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(nms_threshold=self.config.RPN_NMS_THRESHOLD, name="ROI", config=self.config)([rpn_class, rpn_bbox, anchors, proposal_count])
        

        if training:
            def get_gt_idx(gt_cls_ids):
                positive_cls_ids = tf.gather(gt_cls_ids, tf.where(gt_cls_ids>=0))
                positive_cls_ids = tf.squeeze(positive_cls_ids, 1)
                unique_cls_ids,unique_cls_idx  = tf.unique(positive_cls_ids)
                selected_cls_id = tf.random.shuffle(unique_cls_ids)[0]
                selected_indices = tf.where(gt_cls_ids==selected_cls_id)
                selected_idx = tf.random.shuffle(selected_indices)[0]
                return tf.cast(selected_idx, tf.int32)
            idx = tf.squeeze(tf.map_fn(get_gt_idx, input_gt_class_ids), 1)
            prn_cls = tf.squeeze(tf.gather(input_gt_class_ids, idx, axis=1), 1)
            bboxes = tf.squeeze(tf.gather(input_gt_boxes, idx, axis=1), 1)
            gt_masks = tf.squeeze(tf.gather(input_gt_masks, idx, axis=3), 3)
            gt_masks = tf.cast(gt_masks,tf.float32)
            gt_masks = tf.expand_dims(gt_masks,3)

            def make_prn_maks(gt_mask, bbox):
                y1 = bbox[0]
                x1 = bbox[1]
                y2 = bbox[2]
                x2 = bbox[3]
                h = y2-y1
                w = x2-x1
                prn_mask = tf.zeros([*input_image.get_shape()[1:3], 1], tf.float32)
                def _make_prn_maks(prn_mask, gt_mask):
                    gt_mask = tf.image.resize(gt_mask,[h,w])
                    prn_ys, prn_xs = tf.meshgrid(tf.range(y1,y2),tf.range(x1,x2))
                    gt_ys, gt_xs = tf.meshgrid(tf.range(h),tf.range(w))
                    prn_indices = tf.transpose([tf.reshape(prn_ys, [-1]),tf.reshape(prn_xs, [-1])])
                    gt_indices = tf.transpose([tf.reshape(gt_ys, [-1]),tf.reshape(gt_xs, [-1])])
                    prn_mask = tf.tensor_scatter_nd_update(prn_mask, prn_indices, tf.gather_nd(gt_mask, gt_indices))
                    return prn_mask
                prn_mask = tf.cond(h==0 or w==0,lambda: prn_mask, lambda: _make_prn_maks(prn_mask, gt_mask))
                return prn_mask

            prn_masks = tf.map_fn(lambda inputs: make_prn_maks(inputs[0], inputs[1]), [gt_masks, bboxes],
                                fn_output_signature=tf.TensorSpec(shape=(*input_image.get_shape()[1:3], 1), dtype=tf.float32))
            prn_masks = tf.image.resize(prn_masks,[224,224])
            image = tf.image.resize(input_image,[224,224])
            input_prn = tf.concat([image, prn_masks], 3)

            x = KL.ZeroPadding2D(padding=(3,3))(input_prn)
            x = self.meta_conv1(x)
            prn_P2, prn_P3, prn_P4, prn_P5, prn_P6 = self.backbone2fpn(x)

            Gavg_P2 = KL.GlobalAvgPool2D()(prn_P2)
            Gavg_P3 = KL.GlobalAvgPool2D()(prn_P3)
            Gavg_P4 = KL.GlobalAvgPool2D()(prn_P4)
            Gavg_P5 = KL.GlobalAvgPool2D()(prn_P5)
            Gavg_P6 = KL.GlobalAvgPool2D()(prn_P6)

            Gavg = tf.reduce_mean(tf.stack([Gavg_P2,Gavg_P3,Gavg_P4,Gavg_P5,Gavg_P6]), 0)
            attentions = KL.Activation(tf.nn.sigmoid)(Gavg)
            meta_score = self.meta_cls_score(attentions)
            attentions = tf.reshape(attentions,[attentions.get_shape()[0],1,1,self.config.TOP_DOWN_PYRAMID_SIZE])

            # Normalize coordinates
            gt_boxes = norm_boxes_graph(tf.cast(input_gt_boxes,tf.float32), input_image.shape[1:3])

            if not self.config.USE_RPN_ROIS:
                # Normalize coordinates
                target_rois = norm_boxes_graph(tf.cast(input_rois,tf.float32), input_image.shape[1:3])
            else:
                target_rois = rpn_rois

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_bbox, target_mask = DetectionTargetLayer(self.config, name="proposal_targets")([target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            roi_cls_feature = self.ROIAlign_classifier([rois, input_image_meta] + mrcnn_feature_maps)
            roi_seg_feature = self.ROIAlign_mask([rois, input_image_meta] + mrcnn_feature_maps)

            attentive_cls_feature = roi_cls_feature * attentions
            attentive_seg_feature = roi_seg_feature * attentions

            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.fpn_classifier(attentive_cls_feature, training=self.config.TRAIN_BN)
            mrcnn_mask = self.fpn_mask(attentive_seg_feature, training=self.config.TRAIN_BN)

            rpn_class_loss = tf.reduce_mean(self.config.LOSS_WEIGHTS.get('rpn_class_loss', 1.)
                                            * rpn_class_loss_graph(input_rpn_match, rpn_class_logits), keepdims=True)
            rpn_bbox_loss = tf.reduce_mean(self.config.LOSS_WEIGHTS.get('rpn_bbox_loss', 1.) 
                                            * rpn_bbox_loss_graph(self.config,input_rpn_bbox, input_rpn_match, rpn_bbox), keepdims=True)
            class_loss = tf.reduce_mean(self.config.LOSS_WEIGHTS.get('mrcnn_class_loss', 1.) 
                                        * mrcnn_class_loss_graph(target_class_ids, mrcnn_class_logits, active_class_ids), keepdims=True)
            bbox_loss = tf.reduce_mean(self.config.LOSS_WEIGHTS.get('mrcnn_bbox_loss', 1.) 
                                        * mrcnn_bbox_loss_graph(target_bbox, target_class_ids, mrcnn_bbox), keepdims=True)
            mask_loss = tf.reduce_mean(self.config.LOSS_WEIGHTS.get('mrcnn_mask_loss', 1.) 
                                        * mrcnn_mask_loss_graph(target_mask, target_class_ids, mrcnn_mask), keepdims=True)
            reg_losses = tf.add_n([keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
                                                for w in self.trainable_weights
                                                if 'gamma' not in w.name and 'beta' not in w.name])
            meta_loss = tf.reduce_mean(self.config.LOSS_WEIGHTS.get('meta_loss', 1.)
                                    * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=prn_cls, logits=meta_score))
            
            self.add_loss(reg_losses)
            self.add_loss(rpn_class_loss)
            self.add_loss(rpn_bbox_loss)
            self.add_loss(class_loss)
            self.add_loss(bbox_loss)
            self.add_loss(mask_loss)
            self.add_loss(meta_loss)

            output = [rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss, reg_losses, meta_loss],\
                    [attentions, prn_cls]

        else:

            # Network Heads
            # Proposal classifier and BBox regressor heads
            def fpn_inference(attentions):
                attentions = tf.reshape(attentions, [1,1,1,1,self.config.TOP_DOWN_PYRAMID_SIZE])

                roi_cls_feature = self.ROIAlign_classifier([rpn_rois, input_image_meta] + mrcnn_feature_maps)
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.fpn_classifier(roi_cls_feature, training=False)

                # Detections
                # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in
                # normalized coordinates
                detections = DetectionLayer(self.config, name="mrcnn_detection")([rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

                # Create masks for detections
                detection_boxes = detections[..., :4]
                roi_seg_feature = self.ROIAlign_mask([detection_boxes, input_image_meta] + mrcnn_feature_maps)
                mrcnn_mask = self.fpn_mask(roi_seg_feature, training=False)
                return detections, mrcnn_mask

            detections, mrcnn_mask = \
                tf.map_fn(fpn_inference, attentions, fn_output_signature=(tf.TensorSpec(shape=(batch_size, 
                                                                                            self.config.DETECTION_MAX_INSTANCES,
                                                                                            6), 
                                                                                        dtype=tf.float32),
                                                                        tf.TensorSpec(shape=(batch_size,
                                                                                            self.config.DETECTION_MAX_INSTANCES,
                                                                                            self.config.MASK_POOL_SIZE*2,
                                                                                            self.config.MASK_POOL_SIZE*2,
                                                                                            self.config.NUM_CLASSES), 
                                                                                    dtype=tf.float32),
                                                                        )
                        )
            detections = tf.squeeze(tf.concat(tf.split(detections, detections.get_shape()[0]),2),0)
            mrcnn_mask = tf.squeeze(tf.concat(tf.split(mrcnn_mask, mrcnn_mask.get_shape()[0]),2),0)


            output = [detections, mrcnn_mask]

        return output

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
        from pathlib import Path
        if Path(filepath).suffix == '.h5':
            import h5py
            import numpy as np

            f = h5py.File(filepath, mode='r')
            saved_layer_names = [name.decode('utf-8') for name in f.attrs['layer_names']]
            weighted_layers = collect_layers(self)
            for l in weighted_layers:
                layer_name = l.name
                if layer_name in saved_layer_names:
                    sort_key = [w.name.split('/')[-1] for w in l.weights]
                    weights = [np.array(f[f'/{layer_name}/{layer_name}/{name}']) for name in sort_key]
                    l.set_weights(weights)
        else:
            ckpt = tf.train.Checkpoint(model=self)
            manager = tf.train.CheckpointManager(ckpt, directory='save_model', max_to_keep=None)
            status = ckpt.restore(filepath)
        return self
    
    @property
    def is_restore(self):
        return self._restore

def collect_layers(model):
    layers = []
    if isinstance(model, KM.Model):
        for layer in model.layers:
            layers.extend(collect_layers(layer))
    else:
        layers.append(model)
    return layers
