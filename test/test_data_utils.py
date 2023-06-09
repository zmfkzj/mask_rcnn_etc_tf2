import unittest
import cv2
import tensorflow as tf
from MRCNN.data.utils import *
from MRCNN.model.faster_rcnn import FasterRcnn
from MRCNN.utils import denorm_boxes
from MRCNN.layer.proposal import apply_box_deltas_graph

policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)


class TestDataUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.json_path = 'test/source/instances_train_fix.json'
        self.config = Config()
        # self.json_path = 'test/source/test.json'


    def test_load_ann(self):
        dataset = Dataset.from_json(self.json_path,'test/source/images')
        ann = dataset.annotations[0]
        image = load_image(ann.image.path).numpy()

        ann = dataset.coco.loadAnns(ann.id)[0]
        anns = load_ann(dataset, ann, image.shape[:2])
        print()


    def test_box_matching(self):
        dataset = Dataset.from_json(self.json_path,'test/source/images')
        resize_shape = self.config.IMAGE_SHAPE[:2].astype(np.int32)
        mini_mask_shape = self.config.MINI_MASK_SHAPE
        pixel_mean = self.config.PIXEL_MEAN
        pixel_std = self.config.PIXEL_STD
        anchors = get_anchors(self.config)
        rpn_train_anchors_per_image = self.config.RPN_TRAIN_ANCHORS_PER_IMAGE
        rpn_bbox_std_dev = self.config.RPN_BBOX_STD_DEV

        ann = dataset.annotations[0]
        image = load_image(ann.image.path).numpy()
        ann_ids = [ann.id]

        boxes, loader_class_ids, masks =\
            tf.py_function(lambda ann_ids, h, w: load_gt(dataset, ann_ids, h, w), 
                        (ann_ids,tf.shape(image)[0],tf.shape(image)[1]),(tf.float16, tf.int16, tf.bool))

        resized_image, resized_boxes, resized_masks = resize(image, boxes, masks, resize_shape, mini_mask_shape)
        
        norm_image = normalize(resized_image, pixel_mean, pixel_std)

        rpn_match, input_rpn_bbox = build_rpn_targets(loader_class_ids, resized_boxes, anchors,rpn_train_anchors_per_image,rpn_bbox_std_dev)
        batch_counts = tf.reduce_sum(tf.cast(tf.equal(rpn_match, 1), tf.int32))
        input_rpn_bbox = input_rpn_bbox[:batch_counts]

        model = FasterRcnn(self.config,dataset)
        rpn_rois, mrcnn_feature_maps, rpn_class_logits, pred_rpn_bbox = model(tf.expand_dims(norm_image, 0))

        indices = tf.where(tf.equal(rpn_match, 1))
        pred_rpn_bbox = tf.gather_nd(pred_rpn_bbox[0], indices)
        anchors = tf.cast(tf.gather_nd(anchors, indices), tf.float16)

        for i, (a, prb, irb) in enumerate(zip(anchors, pred_rpn_bbox, input_rpn_bbox)):
            a = tf.expand_dims(a, 0)
            prb = tf.expand_dims(prb, 0)
            irb = tf.expand_dims(irb, 0)

            bbox_std_dev = tf.cast(tf.expand_dims(self.config.BBOX_STD_DEV, 0), tf.float16)

            a_ = tuple(a.numpy().round().astype(np.int32)[0,[1,0,3,2]])
            prb = tuple(apply_box_deltas_graph(a, prb * bbox_std_dev).numpy().round().astype(np.int32)[0,[1,0,3,2]])
            irb = tuple(apply_box_deltas_graph(a, irb * bbox_std_dev).numpy().round().astype(np.int32)[0,[1,0,3,2]])
            
            print(a)
            print(prb)
            print(irb)

            img = cv2.cvtColor(resized_image.numpy().copy(), cv2.COLOR_RGB2BGR)
            img = cv2.rectangle(img, a_[:2], a_[2:], color=(255,0,255))
            img = cv2.rectangle(img, prb[:2], prb[2:], color=(0,0,255))
            img = cv2.rectangle(img, irb[:2], irb[2:], color=(255,255,0))

            cv2.imwrite(f'{i}.jpg', img)