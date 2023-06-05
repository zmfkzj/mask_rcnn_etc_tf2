from enum import Enum
from functools import partial
from transformers import TFViTModel, TFConvNextModel
import tfimm
import tensorflow as tf

from MRCNN.config import Config

def build_vit_backbone(pretrained_model, input_size):
    model = TFViTModel.from_pretrained(pretrained_model, add_pooling_layer=False)
    inputs = tf.keras.Input(shape=(*input_size, 3))
    input_x = tf.transpose(inputs, [0,3,1,2])
    
    logits = model(input_x, interpolate_pos_encoding=True).logits
    
    return tf.keras.Model(inputs=inputs, outputs=logits)


def build_convnext_backbone(pretrained_model, input_size):
    model = TFConvNextModel.from_pretrained(pretrained_model, add_pooling_layer=False)
    inputs = tf.keras.Input(shape=(*input_size, 3))
    input_x = tf.transpose(inputs, [0,3,1,2])
    
    logits = model(input_x).logits
    
    return tf.keras.Model(inputs=inputs, outputs=logits)


def build_keras_backbone(model, input_size):
    inputs = 
    outputs = model(inputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs)
    

backbones = {
    'convnext_base': partial(build_convnext_backbone, pretrained_model='facebook/convnext-base-384'),
    'convnext_tiny': partial(build_convnext_backbone, pretrained_model='facebook/convnext-tiny-224'),
    'vit_base_patch16': partial(build_vit_backbone, pretrained_model='google/vit-base-patch16-224'),
    'vit_base_patch32': partial(build_vit_backbone, pretrained_model='google/vit-base-patch32-224-in21k'),
    'vit_large_patch16': partial(build_vit_backbone, pretrained_model='google/vit-large-patch16-224'),
    'vit_large_patch32': partial(build_vit_backbone, pretrained_model='google/vit-large-patch32-224'),
    'ResNet101': tf.keras.applications.ResNet101(include_top=False, pooling='avg', input_tensor=tf.keras.Input(shape=(None,None, 3))),
    'ResNet50V2': tf.keras.applications.ResNet50V2(include_top=False, pooling='avg', input_tensor=tf.keras.Input(shape=(None,None, 3))),
    'ResNet101V2': tf.keras.applications.ResNet101V2(include_top=False, pooling='avg', input_tensor=tf.keras.Input(shape=(None,None, 3))),
    'ResNet152V2': tf.keras.applications.ResNet152V2(include_top=False, pooling='avg', input_tensor=tf.keras.Input(shape=(None,None, 3))),
    'EfficientNetV2B0': tf.keras.applications.EfficientNetV2B0(include_top=False, pooling='avg',include_preprocessing=False, input_tensor=tf.keras.Input(shape=(None,None, 3))),
    'EfficientNetV2B1': tf.keras.applications.EfficientNetV2B1(include_top=False, pooling='avg',include_preprocessing=False, input_tensor=tf.keras.Input(shape=(None,None, 3))),
    'EfficientNetV2B2': tf.keras.applications.EfficientNetV2B2(include_top=False, pooling='avg',include_preprocessing=False, input_tensor=tf.keras.Input(shape=(None,None, 3))),
    'EfficientNetV2B3': tf.keras.applications.EfficientNetV2B3(include_top=False, pooling='avg',include_preprocessing=False, input_tensor=tf.keras.Input(shape=(None,None, 3))),
    'EfficientNetV2L': tf.keras.applications.EfficientNetV2L(include_top=False, pooling='avg',include_preprocessing=False, input_tensor=tf.keras.Input(shape=(None,None, 3))),
    'EfficientNetV2M': tf.keras.applications.EfficientNetV2M(include_top=False, pooling='avg',include_preprocessing=False, input_tensor=tf.keras.Input(shape=(None,None, 3))),
    'EfficientNetV2S': tf.keras.applications.EfficientNetV2S(include_top=False, pooling='avg',include_preprocessing=False, input_tensor=tf.keras.Input(shape=(None,None, 3))),
}