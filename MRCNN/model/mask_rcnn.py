import re
import sys
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM

from MRCNN import utils
from MRCNN.config import Config
from MRCNN.loss import mrcnn_bbox_loss_graph, mrcnn_class_loss_graph, mrcnn_mask_loss_graph, rpn_bbox_loss_graph, rpn_class_loss_graph
from MRCNN.layer.roialign import parse_image_meta_graph
from . import Resnet, FPN_classifier, FPN_mask, RPN
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

        self.backbone = Resnet(config.BACKBONE)

        self.meta_conv1 = KL.Conv2D(64,(7,7),strides=(2,2),name='meta_conv1', use_bias=True)
        self.conv1 = KL.Conv2D(64,(7,7),strides=(2,2),name='conv1', use_bias=True)
        
        self.rpn = RPN(self.config.RPN_ANCHOR_STRIDE, len(self.config.RPN_ANCHOR_RATIOS), name='rpn_model')

        self.ROIAlign_classifier = PyramidROIAlign([config.POOL_SIZE, config.POOL_SIZE], name="roi_align_classifier")
        self.ROIAlign_mask = PyramidROIAlign([config.MASK_POOL_SIZE, config.MASK_POOL_SIZE], name="roi_align_mask")

        self.fpn_classifier = FPN_classifier(self.config.POOL_SIZE, self.config.NUM_CLASSES, fc_layers_size=self.config.FPN_CLASSIF_FC_LAYERS_SIZE)
        self.fpn_mask = FPN_mask(self.config.NUM_CLASSES)

        # Top-down Layers
        # TODO: add assert to varify feature map sizes match what's in config
        self.fpn_c5p5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')
        self.fpn_c4p4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')
        self.fpn_c3p3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')
        self.fpn_c2p2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')
        # Attach 3x3 conv to all P layers to get the final feature maps.
        self.fpn_p2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")
        self.fpn_p3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")
        self.fpn_p4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")
        self.fpn_p5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")

        self.concats = [KL.Concatenate(axis=1, name=n) for n in ["rpn_class_logits", "rpn_class", "rpn_bbox"]]

        self.anchors = self.get_anchors(self.config.IMAGE_SHAPE)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?

        with self.config.STRATEGY.scope():
            inputs = (tf.zeros([1,512,512,3]), tf.zeros([1,20]))
            attention = tf.zeros([256])
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
        _, C2, C3, C4, C5 = self.backbone(x, training=self.config.TRAIN_BN)

        # Top-down Layers
        # TODO: add assert to varify feature map sizes match what's in config
        P5 = self.fpn_c5p5(C5)
        P4 = KL.Add(name="fpn_p4add")([ KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
                                        self.fpn_c4p4(C4)])
        P3 = KL.Add(name="fpn_p3add")([ KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
                                        self.fpn_c3p3(C3)])
        P2 = KL.Add(name="fpn_p2add")([ KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
                                        self.fpn_c2p2(C2)])
        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = self.fpn_p2(P2)
        P3 = self.fpn_p3(P3)
        P4 = self.fpn_p4(P4)
        P5 = self.fpn_p5(P5)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P4)

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        # make prn input image
        def make_prn_features(gt_mask):
            gt_mask = tf.cast(gt_mask,tf.float32)
            gt_mask = tf.expand_dims(gt_mask,3)
            gt_mask = tf.image.resize(gt_mask,[224,224])
            image = tf.image.resize(input_image,[224,224])
            input_prn = tf.concat([image, gt_mask], 3)
            x = KL.ZeroPadding2D(padding=(3,3))(input_prn)
            x = self.meta_conv1(x)
            _, prn_C2, prn_C3, prn_C4, prn_C5 = self.backbone(x, training=self.config.TRAIN_BN)
            prn_P5 = self.fpn_c5p5(prn_C5)
            prn_P4 = KL.Add(name="fpn_p4add")([ KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(prn_P5),
                                            self.fpn_c4p4(prn_C4)])
            prn_P3 = KL.Add(name="fpn_p3add")([ KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(prn_P4),
                                            self.fpn_c3p3(prn_C3)])
            prn_P2 = KL.Add(name="fpn_p2add")([ KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(prn_P3),
                                            self.fpn_c2p2(prn_C2)])
            # Attach 3x3 conv to all P layers to get the final feature maps.
            prn_P2 = self.fpn_p2(prn_P2)
            prn_P3 = self.fpn_p3(prn_P3)
            prn_P4 = self.fpn_p4(prn_P4)
            prn_P5 = self.fpn_p5(prn_P5)

            attention_P2 = KL.Activation(tf.nn.sigmoid)(KL.GlobalAvgPool2D()(prn_P2))
            attention_P3 = KL.Activation(tf.nn.sigmoid)(KL.GlobalAvgPool2D()(prn_P3))
            attention_P4 = KL.Activation(tf.nn.sigmoid)(KL.GlobalAvgPool2D()(prn_P4))
            attention_P5 = KL.Activation(tf.nn.sigmoid)(KL.GlobalAvgPool2D()(prn_P5))

            attention = tf.stack([attention_P2, attention_P3, attention_P4, attention_P5],0)
            attention = tf.math.reduce_mean(attention,0)
            return attention


        if attentions is None and training:
            attentions = tf.map_fn(lambda mask: make_prn_features(mask),
                                    tf.transpose(input_gt_masks, [3,0,1,2]),
                                    fn_output_signature=tf.TensorSpec(shape=(batch_size,self.config.TOP_DOWN_PYRAMID_SIZE,), 
                                                                    dtype=tf.float32)
                                    )
            attentions = tf.reshape(attentions, [tf.shape(attentions)[0], batch_size,1,1,1,self.config.TOP_DOWN_PYRAMID_SIZE])

        # # Anchors
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

            # Network Heads
            # TODO: verify that this handles zero padded ROIs
            def forward_fpn(attention):
                attentive_cls_feature = roi_cls_feature * attention
                attentive_seg_feature = roi_seg_feature * attention
                tf.print(tf.shape(attentive_seg_feature))
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
                return [rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss, reg_losses]
            
            rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss, reg_losses = \
                tf.map_fn(forward_fpn, attentions, fn_output_signature=[tf.TensorSpec(shape=(), dtype=tf.float32),
                                                                        tf.TensorSpec(shape=(), dtype=tf.float32),
                                                                        tf.TensorSpec(shape=(), dtype=tf.float32),
                                                                        tf.TensorSpec(shape=(), dtype=tf.float32),
                                                                        tf.TensorSpec(shape=(), dtype=tf.float32),
                                                                        tf.TensorSpec(shape=(), dtype=tf.float32),
                                                                        ]
                        ,swap_memory=True)

            reg_losses = tf.reduce_mean(reg_losses)
            rpn_class_loss = tf.reduce_mean(rpn_class_loss)
            rpn_bbox_loss = tf.reduce_mean(rpn_bbox_loss)
            class_loss = tf.reduce_mean(class_loss)
            bbox_loss = tf.reduce_mean(bbox_loss)
            mask_loss = tf.reduce_mean(mask_loss)

            self.add_loss(reg_losses)
            self.add_loss(rpn_class_loss)
            self.add_loss(rpn_bbox_loss)
            self.add_loss(class_loss)
            self.add_loss(bbox_loss)
            self.add_loss(mask_loss)

            output = [rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss, reg_losses]

        else:

            # Network Heads
            # Proposal classifier and BBox regressor heads
            roi_cls_feature = self.ROIAlign_classifier([rpn_rois, input_image_meta] + mrcnn_feature_maps)
            attentive_cls_feature = roi_cls_feature * attentions
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.fpn_classifier(attentive_cls_feature, training=False)

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in
            # normalized coordinates
            detections = DetectionLayer(self.config, name="mrcnn_detection")([rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

            # Create masks for detections
            detection_boxes = detections[..., :4]
            roi_seg_feature = self.ROIAlign_mask([detection_boxes, input_image_meta] + mrcnn_feature_maps)
            attentive_seg_feature = roi_seg_feature * attentions
            mrcnn_mask = self.fpn_mask(attentive_seg_feature, training=False)

            output = [detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, rpn_rois, rpn_class, rpn_bbox]

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
