from MRCNN.data_loader import CocoDataset
from MRCNN.model.mask_rcnn import MaskRCNN
from MRCNN.config import Config
from MRCNN.train import Trainer

class CustomConfig(Config):
    GPUS = 0,1
    NUM_CLASSES = 1+80 

train_dataset = CocoDataset()
train_dataset.load_coco()

config = CustomConfig()
model = MaskRCNN(config)

trainer = Trainer(model, train_dataset, config=config)
trainer.train(50, 'all')

