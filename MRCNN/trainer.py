import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as KL

from tqdm import tqdm

from MRCNN.model.mask_rcnn import MaskRCNN
from MRCNN.config import Config
from MRCNN.loss import mrcnn_bbox_loss_graph, mrcnn_class_loss_graph, mrcnn_mask_loss_graph, rpn_bbox_loss_graph, rpn_class_loss_graph
from MRCNN.data.data_generator import data_generator
from MRCNN.layer.roialign import parse_image_meta_graph
from MRCNN.evaluator import Evaluator


class Trainer:
    def __init__(self, model:MaskRCNN, dataset, 
                        val_evaluator:Evaluator = None, test_evaluator:Evaluator = None,
                        optimizer = keras.optimizers.SGD(),
                        config:Config = Config(),
                        augment = False,
                        augmentation = None,
                        logs_dir='logs/'):
        self.config = config
        self.val_evaluator = val_evaluator
        self.test_evaluator = test_evaluator

        self.summary_writer = tf.summary.create_file_writer(logs_dir)
        dataset.prepare()
        
        self.mirrored_strategy = config.MIRRORED_STRATEGY

        with self.mirrored_strategy.scope():
            self.optimizer = optimizer
            self.model = model
            self.dataset = self.mirrored_strategy.experimental_distribute_dataset(
                                    tf.data.Dataset.from_generator(lambda : data_generator(dataset, config, augment=augment, augmentation=augmentation, batch_size=config.BATCH_SIZE),
                                                    output_types=(((tf.float32, tf.float64, tf.int32, tf.float64, tf.int32, tf.int32, tf.bool),()))).prefetch(tf.data.AUTOTUNE))

    def train(self, max_epoch, layers):
        assert layers in ['heads','3+','4+','5+','all']

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        self.model.set_trainable(layers)
        
        ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        manager = tf.train.CheckpointManager(ckpt, directory='save_model', max_to_keep=None)
        status = ckpt.restore(manager.latest_checkpoint)

        for epoch in range(max_epoch):
            pbar = tqdm(desc=f'Epoch : {epoch+1}/{max_epoch}',unit='step', total = self.config.STEPS_PER_EPOCH)
            with self.mirrored_strategy.scope():
                for i, (inputs, _) in enumerate(self.dataset):
                    if self.config.STEPS_PER_EPOCH > i:
                        mean_loss = self.train_step(inputs)
                    else:
                        break

                    pbar.update()
                    pbar.set_postfix({'mean_loss':mean_loss.numpy()})
            pbar.close()
            manager.save()

    @tf.function
    def train_step(self, dist_inputs):
        images, input_image_meta, \
        input_rpn_match, input_rpn_bbox, \
        input_gt_class_ids, input_gt_boxes, input_gt_masks = dist_inputs
        def step_fn(images, input_image_meta, 
                    input_rpn_match, input_rpn_bbox, 
                    input_gt_class_ids, input_gt_boxes, input_gt_masks):
            with tf.GradientTape() as tape:
                output = self.model(images, input_image_meta, 
                                    input_anchors=None, 
                                    input_gt_class_ids=input_gt_class_ids, 
                                    input_gt_boxes=input_gt_boxes, 
                                    input_gt_masks=input_gt_masks, 
                                    input_rois = None,
                                    training=True)
                active_class_ids = KL.Lambda( lambda x: parse_image_meta_graph(x)["active_class_ids"])(input_image_meta)
                loss = self.cal_loss(active_class_ids, input_rpn_match, input_rpn_bbox,*output) / self.config.BATCH_SIZE
            
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(list(zip(grads, self.model.trainable_variables)))
            return loss
        
        per_example_losses = self.mirrored_strategy.run(step_fn, args=(images, input_image_meta, 
                                                                    input_rpn_match, input_rpn_bbox, 
                                                                    input_gt_class_ids, input_gt_boxes, input_gt_masks))
        mean_loss = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_example_losses,axis=None)
        return mean_loss
    
    def cal_loss(self, active_class_ids, input_rpn_match, input_rpn_bbox, 
                        rpn_class_logits, rpn_bbox, target_class_ids, mrcnn_class_logits, target_bbox, mrcnn_bbox, target_mask, mrcnn_mask):

        rpn_class_loss = tf.reduce_mean(self.config.LOSS_WEIGHTS.get('rpn_class_loss', 1.)
                                        * KL.Lambda(lambda x: rpn_class_loss_graph(*x))([input_rpn_match, rpn_class_logits]), keepdims=True)
        rpn_bbox_loss = tf.reduce_mean(self.config.LOSS_WEIGHTS.get('rpn_bbox_loss', 1.) 
                                        * KL.Lambda(lambda x: rpn_bbox_loss_graph(self.config,*x))([input_rpn_bbox, input_rpn_match, rpn_bbox]), keepdims=True)
        class_loss = tf.reduce_mean(self.config.LOSS_WEIGHTS.get('mrcnn_class_loss', 1.) 
                                    * KL.Lambda(lambda x: mrcnn_class_loss_graph(*x))([target_class_ids, mrcnn_class_logits, active_class_ids]), keepdims=True)
        bbox_loss = tf.reduce_mean(self.config.LOSS_WEIGHTS.get('mrcnn_bbox_loss', 1.) 
                                    * KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x))([target_bbox, target_class_ids, mrcnn_bbox]), keepdims=True)
        mask_loss = tf.reduce_mean(self.config.LOSS_WEIGHTS.get('mrcnn_mask_loss', 1.) 
                                    * KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x))([target_mask, target_class_ids, mrcnn_mask]), keepdims=True)
        
        reg_losses = tf.add_n([keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
                                            for w in self.model.trainable_weights
                                            if 'gamma' not in w.name and 'beta' not in w.name])

        total_loss = tf.reduce_sum([rpn_bbox_loss, rpn_class_loss, class_loss, bbox_loss, mask_loss, reg_losses])

        with self.summary_writer.as_default():
            tf.summary.scalar('rpn_class_loss', rpn_class_loss, step=self.optimizer.iterations)
            tf.summary.scalar('rpn_bbox_loss', rpn_bbox_loss, step=self.optimizer.iterations)
            tf.summary.scalar('mrcnn_class_loss', class_loss, step=self.optimizer.iterations)
            tf.summary.scalar('mrcnn_bbox_loss', bbox_loss, step=self.optimizer.iterations)
            tf.summary.scalar('mrcnn_mask_loss', mask_loss, step=self.optimizer.iterations)
            tf.summary.scalar('reg_losses', reg_losses, step=self.optimizer.iterations)
            tf.summary.scalar('total_loss', total_loss, step=self.optimizer.iterations)

        return total_loss