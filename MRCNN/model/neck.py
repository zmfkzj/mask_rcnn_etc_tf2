import tensorflow as tf
import keras.api._v2.keras.models as KM
import keras.api._v2.keras.layers as KL


class Neck(KM.Model):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        
        # Top-down Layers
        # TODO: add assert to varify feature map sizes match what's in config
        self.fpn_c4p4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')
        self.fpn_c3p3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')
        self.fpn_c2p2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')
        # Attach 3x3 conv to all P layers to get the final feature maps.
        self.fpn_p2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")
        self.fpn_p3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")
        self.fpn_p4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")
        self.fpn_p5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")

        self.fpn_p2_bn = KL.BatchNormalization()
        self.fpn_p3_bn = KL.BatchNormalization()
        self.fpn_p4_bn = KL.BatchNormalization()
    
    @tf.function
    def call(self, C2, C3, C4):

        # Top-down Layers
        # TODO: add assert to varify feature map sizes match what's in config
        P4 = self.fpn_c4p4(C4)
        P3 = KL.Add(name="fpn_p3add")([ KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
                                        self.fpn_c3p3(C3)])
        P2 = KL.Add(name="fpn_p2add")([ KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
                                        self.fpn_c2p2(C2)])
        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = self.fpn_p2(P2)
        # P2 = self.fpn_p2_bn(P2)
        # P2 = KL.ReLU()(P2)

        P3 = self.fpn_p3(P3)
        # P3 = self.fpn_p3_bn(P3)
        # P3 = KL.ReLU()(P3)

        P4 = self.fpn_p4(P4)
        # P4 = self.fpn_p4_bn(P4)
        # P4 = KL.ReLU()(P4)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        P5 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p5")(P4)

        return [P2, P3, P4, P5]
