import tensorflow as tf
import tensorflow.keras as keras

from MRCNN.data_loader import CocoDataset
from MRCNN.model.mask_rcnn import MaskRCNN
from MRCNN.config import Config
from MRCNN.train import Trainer

class CustomConfig(Config):
    # GPUS = 0,1
    GPUS = 0
    NUM_CLASSES = 1+80 
    LEARNING_RATE = 0.0001

train_dataset = CocoDataset()
train_dataset.load_coco('/Users/mkkim/dataset/train2017/', '/Users/mkkim/dataset/annotations/instances_train2017.json')

val_dataset = CocoDataset()
val_dataset.load_coco('/Users/mkkim/dataset/val2017/', '/Users/mkkim/dataset/annotations/instances_val2017.json')

config = CustomConfig()
model = MaskRCNN(config)

optimizer = keras.optimizers.Adam(learning_rate=config.LEARNING_RATE,)
trainer = Trainer(model, train_dataset, config=config, optimizer=optimizer)
trainer.train(50, 'all')

