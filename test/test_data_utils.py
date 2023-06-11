import unittest
import cv2
import tensorflow as tf
from MRCNN.data.utils import *
from MRCNN.enums import TrainLayers
from MRCNN.loss import RpnBboxLossGraph, RpnClassLossGraph
from MRCNN.model.faster_rcnn import FasterRcnn
from MRCNN.utils import denorm_boxes
from MRCNN.layer.proposal import apply_box_deltas_graph
import sys
sys.setrecursionlimit(10**6)

policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)


class TestDataUtils(unittest.TestCase):
    def setUp(self) -> None:
        # self.json_path = 'test/source/instances_train_fix.json'
        self.config = Config()
        # self.json_path = 'test/source/test.json'
        self.json_path = '/home/min/4t/dataset/coco/annotations/instances_val2017.json'


    def test_load_ann(self):
        dataset = Dataset.from_json(self.json_path,'test/source/images')
        ann = dataset.annotations[0]
        image = load_image(ann.image.path).numpy()

        ann = dataset.coco.loadAnns(ann.id)[0]
        anns = load_ann(dataset, ann, image.shape[:2])
        print()


    def test_box_matching(self):
        dataset = Dataset.from_json(self.json_path,'/home/min/4t/dataset/coco/val2017/')
        resize_shape = self.config.IMAGE_SHAPE[:2].astype(np.int32)
        mini_mask_shape = self.config.MINI_MASK_SHAPE
        pixel_mean = self.config.PIXEL_MEAN
        pixel_std = self.config.PIXEL_STD
        anchors = get_anchors(self.config)
        rpn_train_anchors_per_image = self.config.RPN_TRAIN_ANCHORS_PER_IMAGE
        rpn_bbox_std_dev = self.config.RPN_BBOX_STD_DEV
        max_gt_instances = self.config.MAX_GT_INSTANCES
        batch_size = 3

        images = dataset.images[0:batch_size]

        batch_resized_images = []
        batch_norm_images = []
        batch_rpn_match = []
        batch_rpn_bbox = []

        for img in images:
            anns = img.annotations
            ann_ids = [ann.id for ann in anns]

            image = load_image(img.path)
            resized_image, _ = resize_image(image, resize_shape)

            norm_image, resized_boxes, loader_class_ids, rpn_match, input_rpn_bbox, masks = \
                preprocessing_train(img.path, resize_shape, pixel_mean, pixel_std, ann_ids, anchors, rpn_train_anchors_per_image, rpn_bbox_std_dev, max_gt_instances,mini_mask_shape, dataset)

            batch_resized_images.append(resized_image)
            batch_norm_images.append(norm_image)
            batch_rpn_match.append(rpn_match)
            batch_rpn_bbox.append(input_rpn_bbox)


        batch_resized_images = tf.stack(batch_resized_images)
        batch_norm_images  = tf.stack(batch_norm_images)
        batch_rpn_match    = tf.stack(batch_rpn_match)
        batch_rpn_bbox     = tf.stack(batch_rpn_bbox)
        
        model = FasterRcnn(self.config,dataset)
        model.set_trainable(TrainLayers.HEADS)
        optimizer = tf.keras.optimizers.Adam(0.0001)

        for j in range(5):
            for _ in range(3):
                with tf.GradientTape() as tape:
                    rpn_rois, mrcnn_feature_maps, rpn_class_logits, pred_rpn_bbox = model(batch_norm_images)
                    rpn_bbox_loss = RpnBboxLossGraph()(batch_rpn_bbox, batch_rpn_match, pred_rpn_bbox, batch_size)
                    rpn_class_loss = RpnClassLossGraph()(batch_rpn_match, rpn_class_logits)
                grad = tape.gradient([rpn_bbox_loss, rpn_class_loss], model.trainable_variables)
                optimizer.apply_gradients(zip(grad, model.trainable_variables))

            print(rpn_bbox_loss)
            for b in range(batch_size):
                indices = tf.where(tf.equal(batch_rpn_match[b], 1))
                print(pred_rpn_bbox[b].shape)
                print(anchors.shape)
                _pred_rpn_bbox = tf.gather_nd(pred_rpn_bbox[b], indices)
                _anchors = tf.cast(tf.gather_nd(anchors, indices), tf.float16)

                for i, (a, prb, irb) in enumerate(zip(_anchors, _pred_rpn_bbox, batch_rpn_bbox[b])):
                    a = tf.expand_dims(a, 0)
                    prb = tf.expand_dims(prb, 0)
                    irb = tf.expand_dims(irb, 0)

                    bbox_std_dev = tf.cast(tf.expand_dims(self.config.BBOX_STD_DEV, 0), tf.float16)

                    a_ = tuple(a.numpy().round().astype(np.int32)[0,[1,0,3,2]])
                    prb = tuple(apply_box_deltas_graph(a, prb * bbox_std_dev).numpy().round().astype(np.int32)[0,[1,0,3,2]])
                    irb = tuple(apply_box_deltas_graph(a, irb * bbox_std_dev).numpy().round().astype(np.int32)[0,[1,0,3,2]])
                    
                    # print('anchors', a_)
                    # print('pred', prb)
                    # print('gt', irb)

                    img = cv2.cvtColor(batch_resized_images[b].numpy().copy(), cv2.COLOR_RGB2BGR)
                    img = cv2.rectangle(img, a_[:2], a_[2:], color=(255,0,255))
                    img = cv2.rectangle(img, prb[:2], prb[2:], color=(0,0,255))
                    img = cv2.rectangle(img, irb[:2], irb[2:], color=(255,255,0))

                    cv2.imwrite(f'{b}_{i}_{j}.jpg', img)