import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM

class Identity_block(KM.Model):
    def __init__(self, filters, stage, block, use_bias=True):
        super().__init__()
        self.filters = filters
        self.stage = stage
        self.block = block
        self.use_bias = use_bias

    def call(self, input_tensor, train_bn=True):
        nb_filter1, nb_filter2, nb_filter3 = self.filters
        conv_name_base = f'res{self.stage}{self.block}_branch'
        bn_name_base = f'bn{self.stage}{self.block}_branch'

        x = KL.Conv2D(nb_filter1, (1,1), name=f'{conv_name_base}2a', use_bias=self.use_bias)(input_tensor)
        x = KL.BatchNormalization(name=f'{bn_name_base}2a')(x, training=train_bn)
        x = KL.ReLU()(x)

        x = KL.Conv2D(nb_filter2, (3,3), padding='same', name=f'{conv_name_base}2b', use_bias=self.use_bias)(x)
        x = KL.BatchNormalization(name=f'{bn_name_base}2b')(x, training=train_bn)
        x = KL.ReLU()(x)

        x = KL.Conv2D(nb_filter3, (1,1), name=f'{conv_name_base}2c', use_bias=self.use_bias)(x)
        x = KL.BatchNormalization(name=f'{bn_name_base}2c')(x, training=train_bn)
        x = KL.ReLU()(x)

        x = KL.Add()([x,input_tensor])
        x = KL.ReLU(name=f'res{self.stage}{self.block}_out')(x)
        return x

class Conv_block(KM.Model):
    def __init__(self, filters, stage, block, strides=(2,2), use_bias=True):
        super().__init__()
        self.filters = filters
        self.stage = stage
        self.block = block
        self.strides=strides
        self.use_bias=use_bias 

    def call(self, input_tensor, train_bn=True):
        nb_filter1, nb_filter2, nb_filter3 = self.filters
        conv_name_base = f'res{self.stage}{self.block}_branch'
        bn_name_base = f'bn{self.stage}{self.block}_branch'

        x = KL.Conv2D(nb_filter1, (1,1), strides=self.strides, name=f'{conv_name_base}2a', use_bias=self.use_bias)(input_tensor)
        x = KL.BatchNormalization(name=f'{bn_name_base}2a')(x,traininig=train_bn)
        x = KL.ReLU()(x)

        x = KL.Conv2D(nb_filter2, (3,3), padding='same', name=f'{conv_name_base}2b', use_bias=self.use_bias)(x)
        x = KL.BatchNormalization(name=f'{bn_name_base}2b')(x,traininig=train_bn)
        x = KL.ReLU()(x)

        x = KL.Conv2D(nb_filter3, (1,1), name=f'{conv_name_base}2c', use_bias=self.use_bias)(x)
        x = KL.BatchNormalization(name=f'{bn_name_base}2c')(x,traininig=train_bn)
        x = KL.ReLU()(x)

        shortcut = KL.Conv2D(nb_filter3, (1,1), strides=self.strides, name=f'{conv_name_base}1', use_bias=self.use_bias)(input_tensor)
        shortcut = KL.BatchNormalization(name=f'{bn_name_base}1')(shortcut, training=train_bn)

        x = KL.Add()([x, shortcut])
        x = KL.ReLU(name=f'res{self.stage}{self.block}_out')(x)
        return x

class Resnet_graph(KM.Model):
    def __init__(self, architecture, stage5=False) -> None:
        assert architecture in ['resnet50','resnet101']
        super().__init__()
        self.architecture = architecture
        self.stage5=stage5

    def resnet_graph(self, input_image, train_bn=True):
        #stage1
        x = KL.ZeroPadding2D(padding=(3,3))(input_image)
        x = KL.Conv2D(64,(7,7),strides=(2,2),name='conv1', use_bias=True)(x)
        x = KL.BatchNormalization(name='bn_conv1')(x, training=train_bn)
        x = KL.ReLU()(x)
        C1 = x = KL.MaxPool2D((3,3), strides=(2,2),padding='same')(x)
        #stage2
        x = Conv_block([64,64,256], stage=2, block='a',strides=(1,1), train_bn=train_bn)(x)
        x = Identity_block([64,64,256],stage=2, block='b',train_bn=train_bn)(x)
        C2 = x = Identity_block([64,64,256],stage=2, block='c',train_bn=train_bn)(x)
        #stage3
        x = Conv_block([128, 128, 512], stage=3, block='a', train_bn=train_bn)(x)
        x = Identity_block([128, 128, 512], stage=3, block='b', train_bn=train_bn)(x)
        x = Identity_block([128, 128, 512], stage=3, block='c', train_bn=train_bn)(x)
        C3 = x = Identity_block([128, 128, 512], stage=3, block='d', train_bn=train_bn)(x)
        # Stage 4
        x = Conv_block([256, 256, 1024], stage=4, block='a', train_bn=train_bn)(x)
        block_count = {"resnet50": 5, "resnet101": 22}[self.architecture]
        for i in range(block_count):
            x = Identity_block([256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)(x)
        C4 = x
        # Stage 5
        if self.stage5:
            x = Conv_block([512, 512, 2048], stage=5, block='a', train_bn=train_bn)(x)
            x = Identity_block([512, 512, 2048], stage=5, block='b', train_bn=train_bn)(x)
            C5 = x = Identity_block([512, 512, 2048], stage=5, block='c', train_bn=train_bn)(x)
        else:
            C5 = None
        return [C1, C2, C3, C4, C5]