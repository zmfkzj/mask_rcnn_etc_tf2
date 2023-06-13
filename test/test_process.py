import unittest
import cv2
import tensorflow as tf
from MRCNN.data.frcnn_data_loader import make_train_dataloader
from MRCNN.data.utils import *
from MRCNN.config import Config
from MRCNN.model.faster_rcnn import FasterRcnn
from MRCNN.layer.proposal import apply_box_deltas_graph
import sys

from MRCNN.utils import unmold_detections
sys.setrecursionlimit(10**6)

policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)


class TestProcess(unittest.TestCase):
    def setUp(self) -> None:
        # self.json_path = 'test/source/instances_train_fix.json'
        # self.json_path = 'test/source/test.json'
        # self.img_dir = 'test/source/images'
        self.config = Config(AUGMENTORS = None)
        # self.json_path = '/home/min/4t/dataset/coco/annotations/instances_val2017.json'
        # self.img_dir = '/home/min/4t/dataset/coco/val2017'
        self.json_path = '/home/min/4t/dataset/detection_comp/train.json'
        self.img_dir = '/home/min/4t/dataset/detection_comp/train'


    def test_check_frcnn_train_output(self):
        train_dataset = Dataset.from_json('/home/min/4t/dataset/detection_comp/train.json',
                                          '/home/min/4t/dataset/detection_comp/train')
        val_dataset = Dataset.from_json('/home/min/4t/dataset/detection_comp/val.json',
                                          '/home/min/4t/dataset/detection_comp/train')
        train_loader = make_train_dataloader(train_dataset, self.config, 1)
        val_loader = make_train_dataloader(val_dataset, self.config, 1)

        anchors = get_anchors(self.config)
        bbox_std_dev = tf.cast(tf.expand_dims(self.config.BBOX_STD_DEV, 0), tf.float16)
        model = FasterRcnn(self.config, train_dataset)
        model.load_weights('save_ResNet101_2023-06-13T16:03:05.512310/chpt/fingtune/train_loss')

        loader = tf.data.Dataset.zip((train_loader, val_loader)).take(10)
        for i, data in enumerate(loader):
            for t, d in zip(['train', 'val'],data):
                input_images, input_gt_boxes, input_gt_class_ids, input_rpn_match, input_rpn_bbox, input_gt_masks= d

                rpn_rois, mrcnn_feature_maps, rpn_class_logits, rpn_bbox = model(input_images, training=False)
                # model.forward_train(rpn_rois, mrcnn_feature_maps, input_gt_class_ids, input_gt_boxes)
                detections = model.forward_predict_test(rpn_rois, mrcnn_feature_maps, tf.constant([[0,0,*self.config.IMAGE_SHAPE[:2]]]))

                img = input_images[0].numpy().copy()
                img = img * self.config.PIXEL_STD + self.config.PIXEL_MEAN
                img = cv2.cvtColor(img.round().astype(np.uint8), cv2.COLOR_RGB2BGR)

                indices = tf.where(tf.equal(input_rpn_match[0], 1))
                _pred_rpn_bbox = tf.gather_nd(rpn_bbox[0], indices)
                _anchors = tf.cast(tf.gather_nd(anchors, indices), tf.float16)

                for b in _anchors:
                    b = tf.expand_dims(b, 0)
                    b = tuple(b.numpy().round().astype(np.int32)[0,[1,0,3,2]])
                    img = cv2.rectangle(img, b[:2], b[2:], color=(255,0,255))

                for a, b in zip(_anchors,_pred_rpn_bbox):
                    a = tf.expand_dims(a, 0)
                    b = tf.expand_dims(b, 0)
                    b = tuple(apply_box_deltas_graph(a, b * bbox_std_dev).numpy().round().astype(np.int32)[0,[1,0,3,2]])
                    img = cv2.rectangle(img, b[:2], b[2:], color=(0,0,255))

                for a, b in zip(_anchors, input_rpn_bbox[0]):
                    a = tf.expand_dims(a, 0)
                    b = tf.expand_dims(b, 0)
                    b = tuple(apply_box_deltas_graph(a, b * bbox_std_dev).numpy().round().astype(np.int32)[0,[1,0,3,2]])
                    img = cv2.rectangle(img, b[:2], b[2:], color=(255,0,0))

                final_rois, final_class_ids, final_scores, final_masks =\
                    unmold_detections(detections[0].numpy(), self.config.IMAGE_SHAPE, self.config.IMAGE_SHAPE, np.array([0,0,*self.config.IMAGE_SHAPE[:2]]))
                for b, s in zip(final_rois, final_scores):
                    _img = img.copy()
                    b = b.round().astype(np.int32)[[1,0,3,2]]
                    _img = cv2.rectangle(_img, b[:2], b[2:], color=(0,255,0), thickness=2)
                    img = cv2.addWeighted(img, (1-s), _img, s, 0)
                    
                cv2.imwrite(f'{i}_{t}.jpg', img)