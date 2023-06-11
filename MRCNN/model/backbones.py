from enum import Enum
from functools import partial
from transformers import TFConvNextModel
import tfimm
import tensorflow as tf

from MRCNN.config import Config



def build_convnext_backbone(model_name):
    pretrained_model_name = backbones[model_name][0]
    model = TFConvNextModel.from_pretrained(pretrained_model_name)
    inputs = tf.keras.Input(shape=(None,None, 3))
    input_x = tf.transpose(inputs, [0,3,1,2])

    outputs = model(input_x, output_hidden_states=True)
    outputs = outputs.hidden_states[-4:]
    outputs = [tf.transpose(o,[0,2,3,1]) for o in outputs]

    
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def build_keras_backbone(model_name:str):
    model = backbones[model_name][0]()
    output_layers = []
    for l in model.layers:
        input_shape = l.input_shape
        if isinstance( input_shape, list):
            input_shape = [tuple( [s if s is not None else 1024 for s in shape] ) for shape in input_shape]
        else:
            input_shape = tuple( [s if s is not None else 1024 for s in input_shape] )

        output_shape = l.compute_output_shape(input_shape)
        if isinstance(output_shape, list):
            continue
        if output_shape[1]*2 == input_shape[1]:
            output_layers.append(l.input)
    
    output_layers.append(model.output)
    output_layers = sorted([s.deref() for s in set([o.ref() for o in output_layers])], key=lambda x: x.shape[-1])
    outputs = output_layers[-4:]

    return tf.keras.Model(inputs=model.input, outputs=outputs, name='backbone')

backbones = {
    # 'convnext_base': ('facebook/convnext-base-384',build_convnext_backbone),
    # 'convnext_small': ( 'facebook/convnext-small-224', build_convnext_backbone ),
    # 'convnext_tiny': ( 'facebook/convnext-tiny-224' ,build_convnext_backbone),
    # 'ConvNeXtBase': (lambda: tf.keras.applications.ConvNeXtBase(include_top=False,include_preprocessing=False, input_tensor=tf.keras.Input((None,None,3), dtype=tf.float16)) , build_keras_backbone),
    # 'ConvNeXtLarge': (lambda: tf.keras.applications.ConvNeXtLarge(include_top=False,include_preprocessing=False, input_tensor=tf.keras.Input((None,None,3), dtype=tf.float16)) , build_keras_backbone),
    # 'ConvNeXtSmall': (lambda: tf.keras.applications.ConvNeXtSmall(include_top=False,include_preprocessing=False, input_tensor=tf.keras.Input((None,None,3), dtype=tf.float16)) , build_keras_backbone),
    # 'ConvNeXtTiny': (lambda: tf.keras.applications.ConvNeXtTiny(include_top=False,include_preprocessing=False, input_tensor=tf.keras.Input((None,None,3), dtype=tf.float16)) , build_keras_backbone),
    'ResNet101': (lambda: tf.keras.applications.ResNet101(include_top=False, input_tensor=tf.keras.Input((None,None,3), dtype=tf.float16)), build_keras_backbone),
    'ResNet50V2': (lambda: tf.keras.applications.ResNet50V2(include_top=False, input_tensor=tf.keras.Input((None,None,3), dtype=tf.float16)) , build_keras_backbone),
    'ResNet101V2': (lambda: tf.keras.applications.ResNet101V2(include_top=False, input_tensor=tf.keras.Input((None,None,3), dtype=tf.float16)) , build_keras_backbone),
    'ResNet152V2': (lambda: tf.keras.applications.ResNet152V2(include_top=False, input_tensor=tf.keras.Input((None,None,3), dtype=tf.float16)) , build_keras_backbone),
    'EfficientNetV2B0': (lambda: tf.keras.applications.EfficientNetV2B0(include_top=False,include_preprocessing=False, input_tensor=tf.keras.Input((None,None,3), dtype=tf.float16)) , build_keras_backbone),
    'EfficientNetV2B1': (lambda: tf.keras.applications.EfficientNetV2B1(include_top=False,include_preprocessing=False, input_tensor=tf.keras.Input((None,None,3), dtype=tf.float16)) , build_keras_backbone),
    'EfficientNetV2B2': (lambda: tf.keras.applications.EfficientNetV2B2(include_top=False,include_preprocessing=False, input_tensor=tf.keras.Input((None,None,3), dtype=tf.float16)) , build_keras_backbone),
    'EfficientNetV2B3': (lambda: tf.keras.applications.EfficientNetV2B3(include_top=False,include_preprocessing=False, input_tensor=tf.keras.Input((None,None,3), dtype=tf.float16)) , build_keras_backbone),
    'EfficientNetV2L': (lambda: tf.keras.applications.EfficientNetV2L(include_top=False,include_preprocessing=False, input_tensor=tf.keras.Input((None,None,3), dtype=tf.float16)) , build_keras_backbone),
    'EfficientNetV2M': (lambda: tf.keras.applications.EfficientNetV2M(include_top=False,include_preprocessing=False, input_tensor=tf.keras.Input((None,None,3), dtype=tf.float16)) , build_keras_backbone),
    'EfficientNetV2S': (lambda: tf.keras.applications.EfficientNetV2S(include_top=False,include_preprocessing=False, input_tensor=tf.keras.Input((None,None,3), dtype=tf.float16)) , build_keras_backbone),
    'InceptionV3': (lambda: tf.keras.applications.InceptionV3(include_top=False,include_preprocessing=False, input_tensor=tf.keras.Input((None,None,3), dtype=tf.float16)) , build_keras_backbone),
    'InceptionResNetV2': (lambda: tf.keras.applications.InceptionResNetV2(include_top=False,include_preprocessing=False, input_tensor=tf.keras.Input((None,None,3), dtype=tf.float16)) , build_keras_backbone),
}