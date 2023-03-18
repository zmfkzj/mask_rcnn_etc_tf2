from dataclasses import field
from typing import Callable, Iterable, Optional, Union
import numpy as np
import tensorflow as tf
import keras.api._v2.keras as keras
from pydantic.dataclasses import dataclass
from pydantic import Field


# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

@dataclass(unsafe_hash=True)
class Config:
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    PRN_IMAGE_SIZE:tuple[int,int] = (224,224)
    NOVEL_CLASSES:list[int] = field(default_factory=lambda:[])
    SHOTS:int = 3

    ACTIVE_CLASS_IDS:Optional[list[int]] = None

    # NUMBER OFF GPUs to use. When using only a CPU, this needs to be set to 1.
    GPUS:Union[int, list[int]] = 0

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    TRAIN_IMAGES_PER_GPU:int = 2
    TEST_IMAGES_PER_GPU:int = 4
    PRN_IMAGES_PER_GPU:int = 1

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH:int = 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS:int = 50

    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE:Callable = staticmethod(keras.applications.ResNet101)
    PREPROCESSING:Callable = staticmethod(keras.applications.resnet.preprocess_input)


    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES:list[int] = field(default_factory=lambda: [4, 8, 16, 32, 64])
    # BACKBONE_STRIDES = [4, 8, 16, 32]

    # Size of the fully-connected layers in the classification graph
    FPN_CLASSIF_FC_LAYERS_SIZE:int = 1024

    # Size of the top-down layers used to build the feature pyramid
    TOP_DOWN_PYRAMID_SIZE:int = 256

    # Number of classification classes (including background)
    NUM_CLASSES:int = 81  # Override in sub-classes

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES:list[int] = field(default_factory=lambda:[32, 64, 128, 256, 512])
    # RPN_ANCHOR_SCALES = (32, 96, 288, 608)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS:list[float] = field(default_factory=lambda:[0.5, 1, 2])

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE:int = 1

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD:float = 0.7

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE:int = 256
    
    # ROIs kept after tf.nn.top_k and before non-maximum suppression
    PRE_NMS_LIMIT:int = 6000

    # ROIs kept after non-maximum suppression (training and inference)
    POST_NMS_ROIS_TRAINING:int = 2000
    POST_NMS_ROIS_INFERENCE:int = 1000

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK:bool = True
    MINI_MASK_SHAPE:tuple[int,int] = (56, 56)  # (height, width) of the mini-mask

    # Input image resizinga7c202b9-6da1-44fa-8359-7cfa1c8576f7
    # Generally, use the "square" resizing mode for training and predicting
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    # Available resizing modes:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    # crop:   Picks random crops from the image. First, scales the image based
    #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #         IMAGE_MAX_DIM is not used in this mode.
    IMAGE_RESIZE_MODE:str = "square"
    IMAGE_MIN_DIM:int = 800
    IMAGE_MAX_DIM:int = 1024
    # Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
    # up scaling. For example, if set to 2 then images are scaled up to double
    # the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
    # However, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
    IMAGE_MIN_SCALE:int = 0
    # Number of color channels per image. RGB = 3, grayscale = 1, RGB-D = 4
    # Changing this requires other changes in the code. See the WIKI for more
    # details: https://github.com/matterport/Mask_RCNN/wiki
    IMAGE_CHANNEL_COUNT:int = 3

    # Image mean (RGB)
    MEAN_PIXEL:list[float] = field(default_factory=lambda:[123.7, 116.8, 103.9])

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE:int = 200

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO:float = 0.33

    # Pooled ROIs
    POOL_SIZE:int = 7
    MASK_POOL_SIZE:int = 14

    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    MASK_SHAPE:tuple[int,int] = (28, 28)

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES:int = 100

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV:list[float] = field(default_factory=lambda:[0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV:list[float] = field(default_factory=lambda:[0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    DETECTION_MAX_INSTANCES:int = 100

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE:float = 0.

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD:float = 0.3

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    LEARNING_RATE:float = 0.001
    LEARNING_MOMENTUM:float = 0.9

    # Weight decay regularization
    WEIGHT_DECAY:float = 0.0001

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS:dict[str,float] = field(default_factory=lambda:{
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.,
        'meta_loss': 1.
    })


    # Use RPN ROIs or externally generated ROIs for training
    # Keep this True for most situations. Set to False if you want to train
    # the head branches on ROI generated by code rather than the ROIs from
    # the RPN. For example, to debug the classifier head without having to
    # train the RPN.
    USE_RPN_ROIS:bool = True

    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    TRAIN_BN:bool = False  # Defaulting to False since batch size is often small

    # Gradient norm clipping
    GRADIENT_CLIP_NORM:float = 5.0

    def __post_init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        if isinstance(self.GPUS, int):
            self.GPU_COUNT:int = 1
            gpus = [self.GPUS]
        else:
            self.GPU_COUNT = len(self.GPUS)
            gpus = self.GPUS
        self.STRATEGY:tf.distribute.MirroredStrategy = tf.distribute.MirroredStrategy(devices=[f'/gpu:{gpu_id}' for gpu_id in gpus])
        self.TRAIN_BATCH_SIZE:int = self.TRAIN_IMAGES_PER_GPU * self.GPU_COUNT
        self.TEST_BATCH_SIZE:int = self.TEST_IMAGES_PER_GPU * self.GPU_COUNT
        self.PRN_BATCH_SIZE:int = self.PRN_IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        self.IMAGE_SHAPE:np.ndarray = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, self.IMAGE_CHANNEL_COUNT])
        self.origin_NUM_CLASSES = self.NUM_CLASSES

    
    def set_phase(self, phase):
        if phase==1:
            self.NUM_CLASSES = self.origin_NUM_CLASSES - len(self.NOVEL_CLASSES)
        else:
            self.NUM_CLASSES = self.origin_NUM_CLASSES
