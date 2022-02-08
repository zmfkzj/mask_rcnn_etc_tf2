import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM

class Identity_block(KM.Model):
    def __init__(self, filters, stage, block, use_bias=True):
        super().__init__()
        nb_filter1, nb_filter2, nb_filter3 = filters
        conv_name_base = f'res{stage}{block}_branch'
        bn_name_base = f'bn{stage}{block}_branch'

        self.res_branch2a = KL.Conv2D(nb_filter1, (1,1), name=f'{conv_name_base}2a', use_bias=use_bias)
        self.bn_branch2a = KL.BatchNormalization(name=f'{bn_name_base}2a')

        self.res_branch2b = KL.Conv2D(nb_filter2, (3,3), padding='same', name=f'{conv_name_base}2b', use_bias=use_bias)
        self.bn_branch2b = KL.BatchNormalization(name=f'{bn_name_base}2b')

        self.res_branch2c = KL.Conv2D(nb_filter3, (1,1), name=f'{conv_name_base}2c', use_bias=use_bias)
        self.bn_branch2c = KL.BatchNormalization(name=f'{bn_name_base}2c')

        self.res_out = KL.ReLU(name=f'res{stage}{block}_out')

    def call(self, input_tensor, train_bn=True):

        x = self.res_branch2a(input_tensor)
        x = self.bn_branch2a(x, training=train_bn)
        x = KL.ReLU()(x)

        x = self.res_branch2b(x)
        x = self.bn_branch2b(x, training=train_bn)
        x = KL.ReLU()(x)

        x = self.res_branch2c(x)
        x = self.bn_branch2c(x, training=train_bn)
        x = KL.ReLU()(x)

        x = KL.Add()([x,input_tensor])
        x = self.res_out(x)
        return x

class Conv_block(KM.Model):
    def __init__(self, filters, stage, block, strides=(2,2), use_bias=True):
        super().__init__()

        nb_filter1, nb_filter2, nb_filter3 = filters
        conv_name_base = f'res{stage}{block}_branch'
        bn_name_base = f'bn{stage}{block}_branch'

        self.res_branch2a = KL.Conv2D(nb_filter1, (1,1), strides=strides, name=f'{conv_name_base}2a', use_bias=use_bias)
        self.bn_branch2a = KL.BatchNormalization(name=f'{bn_name_base}2a')

        self.res_branch2b = KL.Conv2D(nb_filter2, (3,3), padding='same', name=f'{conv_name_base}2b', use_bias=use_bias)
        self.bn_branch2b = KL.BatchNormalization(name=f'{bn_name_base}2b')

        self.res_branch2c = KL.Conv2D(nb_filter3, (1,1), name=f'{conv_name_base}2c', use_bias=use_bias)(x)
        self.bn_branch2c = KL.BatchNormalization(name=f'{bn_name_base}2c')

        self.res_branch1 = KL.Conv2D(nb_filter3, (1,1), strides=strides, name=f'{conv_name_base}1', use_bias=use_bias)
        self.bn_branch1 = KL.BatchNormalization(name=f'{bn_name_base}1')

        self.res_out = KL.ReLU(name=f'res{stage}{block}_out')

    def call(self, input_tensor, train_bn=True):

        x = self.res_branch2a(input_tensor)
        x = self.bn_branch2a(x,traininig=train_bn)
        x = KL.ReLU()(x)

        x = self.res_branch2b(x)
        x = self.bn_branch2b(x,traininig=train_bn)
        x = KL.ReLU()(x)

        x = self.res_branch2c(x)
        x = self.bn_branch2c(x,traininig=train_bn)
        x = KL.ReLU()(x)

        shortcut = self.res_branch1(input_tensor)
        shortcut = self.bn_branch1(shortcut, training=train_bn)

        x = KL.Add()([x, shortcut])
        x = self.res_out(x)
        return x

class Resnet(KM.Model):
    def __init__(self, architecture, stage5=False) -> None:
        assert architecture in ['resnet50','resnet101']
        super().__init__()
        self.stage5 = stage5

        #stage1
        self.stage1_conv = KL.Conv2D(64,(7,7),strides=(2,2),name='conv1', use_bias=True)
        self.stage1_bn = KL.BatchNormalization(name='bn_conv1')
        #stage2
        self.stage2_conv = Conv_block([64,64,256], stage=2, block='a',strides=(1,1))
        self.stage2_identity_1 = Identity_block([64,64,256],stage=2, block='b')
        self.stage2_identity_2 = Identity_block([64,64,256],stage=2, block='c')
        #stage3
        self.stage3_conv = Conv_block([128, 128, 512], stage=3, block='a')
        self.stage3_identity_1 = Identity_block([128, 128, 512], stage=3, block='b')
        self.stage3_identity_2 = Identity_block([128, 128, 512], stage=3, block='c')
        self.stage3_identity_3 = Identity_block([128, 128, 512], stage=3, block='d')
        # Stage 4
        self.stage4_conv = Conv_block([256, 256, 1024], stage=4, block='a')
        block_count = {"resnet50": 5, "resnet101": 22}[architecture]
        self.seq = []
        for i in range(block_count):
            self.seq.append(Identity_block([256, 256, 1024], stage=4, block=chr(98 + i)))
        # Stage 5
        if stage5:
            self.stage5_conv = Conv_block([512, 512, 2048], stage=5, block='a')
            self.stage5_identity_1 = Identity_block([512, 512, 2048], stage=5, block='b')
            self.stage5_identity_2 = Identity_block([512, 512, 2048], stage=5, block='c')

    def resnet_graph(self, input_image, train_bn=True):
        #stage1
        x = KL.ZeroPadding2D(padding=(3,3))(input_image)
        x = self.stage1_conv(x)
        x = self.stage1_bn(x, training=train_bn)
        x = KL.ReLU()(x)
        C1 = x = KL.MaxPool2D((3,3), strides=(2,2),padding='same')(x)
        #stage2
        x = self.stage2_conv(x, train_bn=train_bn)
        x = self.stage2_identity_1(x, train_bn=train_bn)
        C2 = x = self.stage2_identity_2(x, train_bn=train_bn)(x, train_bn=train_bn)
        #stage3
        x = self.stage3_conv(x, train_bn=train_bn)
        x = self.stage3_identity_1(x, train_bn=train_bn)
        x = self.stage3_identity_2(x, train_bn=train_bn)
        C3 = x =self.stage3_identity_3(x, train_bn=train_bn)
        # Stage 4
        x = self.stage4_conv(x)
        for block in self.seq:
            x = block(x, train_bn=train_bn)
        C4 = x
        # Stage 5
        if self.stage5:
            x = self.stage5_conv(x, train_bn=train_bn)
            x = self.stage5_identity_1(x, train_bn=train_bn)
            C5 = x = self.stage5_identity_2(x, train_bn=train_bn)
        else:
            C5 = None
        return [C1, C2, C3, C4, C5]