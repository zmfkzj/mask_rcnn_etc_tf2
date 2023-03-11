import tensorflow as tf
import keras.api._v2.keras.layers as KL
import keras.api._v2.keras.models as KM
# from ..layer import PyramidROIAlign

class FPN_classifier(KM.Model):
    def __init__(self, pool_size, num_classes, fc_layers_size=1024):
        super().__init__()
        self.pool_size = pool_size
        self.num_classes = num_classes

        # Two 1024 FC layers (implemented with Conv2D for consistency)
        self.timedist_conv_1 = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"), name="mrcnn_class_conv1")
        self.timedist_bn_1 = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_class_bn1')
        self.timedist_conv_2 = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (1, 1)), name="mrcnn_class_conv2")
        self.timedist_bn_2 = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_class_bn2')

        # Classifier head
        self.class_logits = KL.Conv2D(1,(1,1), name='mrcnn_class_logits')
        self.probs = KL.TimeDistributed(KL.Activation("softmax"), name="mrcnn_class")

        # BBox head
        # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
        self.bbox = KL.TimeDistributed(KL.Dense(num_classes * 4, activation='linear'), name='mrcnn_bbox_fc')

        self.mrcnn_bbox = KL.Lambda(lambda t: tf.reshape(t,(tf.shape(t)[0], tf.shape(t)[1], self.num_classes, 4)),name="mrcnn_bbox")

    def call(self, pooled_rois, attentions, training=True):
        """Builds the computation graph of the feature pyramid network classifier
        and regressor heads.

        rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
            coordinates.
        feature_maps: List of feature maps from different layers of the pyramid,
                    [P2, P3, P4, P5]. Each has a different resolution.
        image_meta: [batch, (meta data)] Image details. See compose_image_meta()
        pool_size: The width of the square feature map generated from ROI Pooling.
        num_classes: number of classes, which determines the depth of the results
        train_bn: Boolean. Train or freeze Batch Norm layers
        fc_layers_size: Size of the 2 FC layers

        Returns:
            logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
            probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
            bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to
                        proposal boxes
        """
        # Two 1024 FC layers (implemented with Conv2D for consistency)
        x = self.timedist_conv_1(pooled_rois)
        x = self.timedist_bn_1(x, training=training) #[batch, num_rois,1,1,fc_layers_size]
        x = x * attentions # [batch, num_rois,1,num_classes,fc_layers_size]
        x = KL.Activation('relu')(x)
        x = self.timedist_conv_2(x)
        x = self.timedist_bn_2(x, training=training)
        x = KL.Activation('relu')(x)

        shared = tf.squeeze(x, 2, name='pool_sueeze') #[batch, num_rois,num_classes,fc_layers_size]

        # Classifier head
        mrcnn_class_logits = self.class_logits(shared) #[batch, num_rois,num_classes,1]
        mrcnn_class_logits = tf.squeeze(mrcnn_class_logits, 3) #[batch, num_rois,num_classes]
        mrcnn_probs = self.probs(mrcnn_class_logits)

        # BBox head
        top_score_pos = tf.argmax(mrcnn_class_logits,-1) #[batch, num_rois]
        selected_shared_features = tf.gather(shared,top_score_pos, batch_dims=2)
        # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
        x = self.bbox(selected_shared_features)
        # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
        mrcnn_bbox = self.mrcnn_bbox(x)

        return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


class FPN_mask(KM.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        # Conv layers
        self.timedist_conv_1 = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv1")
        self.timedist_bn_1 = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_mask_bn1')

        self.timedist_conv_2 = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv2")
        self.timedist_bn_2 = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_mask_bn2')

        self.timedist_conv_3 = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv3")
        self.timedist_bn_3 = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_mask_bn3')

        self.timedist_conv_4 = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv4")
        self.timedist_bn_4 = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_mask_bn4')

        self.timedist_convT = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"), name="mrcnn_mask_deconv")
        self.timedist_conv_5 = KL.TimeDistributed(KL.Conv2D(self.num_classes, (1, 1), strides=1, activation="sigmoid"), name="mrcnn_mask")

    def call(self, pooled_rois, training=True):
        """Builds the computation graph of the mask head of Feature Pyramid Network.

        rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
            coordinates.
        feature_maps: List of feature maps from different layers of the pyramid,
                    [P2, P3, P4, P5]. Each has a different resolution.
        image_meta: [batch, (meta data)] Image details. See compose_image_meta()
        pool_size: The width of the square feature map generated from ROI Pooling.
        num_classes: number of classes, which determines the depth of the results
        train_bn: Boolean. Train or freeze Batch Norm layers

        Returns: Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
        """
        # Conv layers
        x = self.timedist_conv_1(pooled_rois)
        x = self.timedist_bn_1(x, training=training)
        x = KL.Activation('relu')(x)

        x = self.timedist_conv_2(x)
        x = self.timedist_bn_2(x, training=training)
        x = KL.Activation('relu')(x)

        x = self.timedist_conv_3(x)
        x = self.timedist_bn_3(x, training=training)
        x = KL.Activation('relu')(x)

        x = self.timedist_conv_4(x)
        x = self.timedist_bn_4(x, training=training)
        x = KL.Activation('relu')(x)

        x = self.timedist_convT(x)
        x = self.timedist_conv_5(x)
        return x