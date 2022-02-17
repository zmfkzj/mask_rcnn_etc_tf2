import os
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
import numpy as np

from tqdm import tqdm

from MRCNN.model.mask_rcnn import MaskRCNN
from MRCNN.config import Config
from MRCNN.loss import mrcnn_bbox_loss_graph, mrcnn_class_loss_graph, mrcnn_mask_loss_graph, rpn_bbox_loss_graph, rpn_class_loss_graph
from MRCNN.data_generator import data_generator
from MRCNN.layer.roialign import parse_image_meta_graph
from MRCNN.evaluation import Evaluator


class Trainer:
    def __init__(self, model:MaskRCNN, train_dataset, 
                        val_dataset = None, test_dataset = None,
                        optimizer = keras.optimizers.SGD(),
                        config = Config(),
                        logs_dir='logs/'):
        self.config = config
        self.summary_writer = tf.summary.create_file_writer(logs_dir)

        if isinstance(config.GPUS, int):
            # self.mirrored_strategy = tf.distribute.MirroredStrategy(devices=[f'/gpu:{config.GPUS}'])
            self.mirrored_strategy = tf.distribute.MirroredStrategy()
        else:
            self.mirrored_strategy = tf.distribute.MirroredStrategy(devices=[f'/gpu:{gpu_id}' for gpu_id in config.GPUS])
        train_dataset = self.load_dataset(train_dataset,subset='train')
        with self.mirrored_strategy.scope():
            self.model = model
            self.train_dataset = self.mirrored_strategy.experimental_distribute_dataset(
                                    tf.data.Dataset.from_generator(lambda : train_dataset,
                                                    output_signature=(((tf.TensorSpec(shape=(config.BATCH_SIZE,config.IMAGE_MAX_DIM,config.IMAGE_MAX_DIM,config.IMAGE_CHANNEL_COUNT), dtype=np.float32),
                                                                    #    tf.RaggedTensorSpec(shape=(config.BATCH_SIZE,None), dtype=tf.float64),
                                                                    #    tf.RaggedTensorSpec(shape=(config.BATCH_SIZE,None, 1), dtype=tf.int32),
                                                                       tf.TensorSpec(shape=(config.BATCH_SIZE,93), dtype=np.float64),
                                                                       tf.TensorSpec(shape=(config.BATCH_SIZE,261888, 1), dtype=np.int32),
                                                                       tf.TensorSpec(shape=(config.BATCH_SIZE,config.RPN_TRAIN_ANCHORS_PER_IMAGE,4), dtype=np.float64),
                                                                       tf.TensorSpec(shape=(config.BATCH_SIZE,config.MAX_GT_INSTANCES), dtype=np.int32),
                                                                       tf.TensorSpec(shape=(config.BATCH_SIZE,config.MAX_GT_INSTANCES,4), dtype=np.int32),
                                                                       tf.TensorSpec(shape=(config.BATCH_SIZE,*config.MINI_MASK_SHAPE,config.MAX_GT_INSTANCES), dtype=np.bool)),
                                                                    ()))).prefetch(tf.data.AUTOTUNE))
            self.optimizer = optimizer

        if val_dataset is not None:
            self.val_dataset = self.load_dataset(val_dataset)
        else:
            self.val_dataset = None
        if test_dataset is not None:
            self.test_dataset = self.load_dataset(test_dataset)
        else:
            self.test_dataset = None

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
        # self.model.build(input_shape=(self.config.BATCH_SIZE,self.config.IMAGE_MAX_DIM,self.config.IMAGE_MAX_DIM,3))
        # self.model.summary()

        ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        manager = tf.train.CheckpointManager(ckpt, directory='save_model', max_to_keep=None)
        status = ckpt.restore(manager.latest_checkpoint)

        for epoch in range(max_epoch):
            pbar = tqdm(desc=f'Epoch : {epoch+1}/{max_epoch}',unit='step', total = self.config.STEPS_PER_EPOCH)
            with self.mirrored_strategy.scope():
                for i, inputs in enumerate(self.train_dataset):
                    if self.config.STEPS_PER_EPOCH >= i:
                        mean_loss = self.train_step(inputs)
                    else:
                        break

                    pbar.update()
                    pbar.set_postfix({'mean_loss':mean_loss.numpy()})
            pbar.close()

            manager.save()

            if self.val_dataset is not None:
                print('Validation Dataset Evaluating...')
                valEval = Evaluator(self.model, self.val_dataset)
                val_recall, val_precision, val_mAP, val_F1 = valEval.evaluate()

                with self.summary_writer.as_default():
                    tf.summary.scalar('val_recall', val_recall, step=epoch)
                    tf.summary.scalar('val_precision', val_precision, step=epoch)
                    tf.summary.scalar('val_mAP', val_mAP, step=epoch)
                    tf.summary.scalar('val_F1', val_F1, step=epoch)

            if self.test_dataset is not None:
                print('Test Dataset Evaluating...')
                testEval = Evaluator(self.model, self.test_dataset)
                test_recall, test_precision, test_mAP, test_F1 = testEval.evaluate()

                with self.summary_writer.as_default():
                    tf.summary.scalar('test_recall', test_recall, step=epoch)
                    tf.summary.scalar('test_precision', test_precision, step=epoch)
                    tf.summary.scalar('test_mAP', test_mAP, step=epoch)
                    tf.summary.scalar('test_F1', test_F1, step=epoch)

    # @tf.function
    def train_step(self, dist_inputs):
        def step_fn(inputs):
            (images, input_image_meta, \
            input_rpn_match, input_rpn_bbox, \
            input_gt_class_ids, input_gt_boxes, input_gt_masks),_ = inputs
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
        
        per_example_losses = self.mirrored_strategy.run(step_fn, args=(dist_inputs,))
        if per_example_losses.shape!=():
            mean_loss = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_example_losses, axis=0)
            return mean_loss
        else:
            return per_example_losses
    
    # @tf.function
    def cal_loss(self, active_class_ids, input_rpn_match, input_rpn_bbox, 
                        rpn_class_logits, rpn_bbox, target_class_ids, mrcnn_class_logits, target_bbox, mrcnn_bbox, target_mask, mrcnn_mask):

        rpn_class_loss = tf.reduce_mean(self.config.LOSS_WEIGHTS.get('rpn_class_loss', 1.)
                                        * rpn_class_loss_graph(*[input_rpn_match, rpn_class_logits]), keepdims=True)
        rpn_bbox_loss = tf.reduce_mean(self.config.LOSS_WEIGHTS.get('rpn_bbox_loss', 1.) 
                                        * rpn_bbox_loss_graph(self.config,*[input_rpn_bbox, input_rpn_match, rpn_bbox]), keepdims=True)
        class_loss = tf.reduce_mean(self.config.LOSS_WEIGHTS.get('mrcnn_class_loss', 1.) 
                                    * mrcnn_class_loss_graph(*[target_class_ids, mrcnn_class_logits, active_class_ids]), keepdims=True)
        bbox_loss = tf.reduce_mean(self.config.LOSS_WEIGHTS.get('mrcnn_bbox_loss', 1.) 
                                    * mrcnn_bbox_loss_graph(*[target_bbox, target_class_ids, mrcnn_bbox]), keepdims=True)
        mask_loss = tf.reduce_mean(self.config.LOSS_WEIGHTS.get('mrcnn_mask_loss', 1.) 
                                    * mrcnn_mask_loss_graph(*[target_mask, target_class_ids, mrcnn_mask]), keepdims=True)
        
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
    
    def load_dataset(self, dataset, augmentation=None, no_augmentation_sources=None, subset='train'):
        assert subset in ['train', 'val', 'test']
        dataset.prepare()
        if subset == 'train':
            return data_generator(dataset, self.config, shuffle=True, augmentation=augmentation, batch_size=self.config.BATCH_SIZE, no_augmentation_sources=no_augmentation_sources)
        else:
            return data_generator(dataset, self.config, shuffle=True, batch_size=self.config.BATCH_SIZE)
    
    def save(self):
        self.model.save()