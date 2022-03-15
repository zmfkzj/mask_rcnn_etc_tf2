import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as KL

from tqdm import tqdm

from .evaluator import Evaluator
from MRCNN.config import Config
from MRCNN.loss import mrcnn_bbox_loss_graph, mrcnn_class_loss_graph, mrcnn_mask_loss_graph, rpn_bbox_loss_graph, rpn_class_loss_graph
from MRCNN.data.data_generator import data_generator


class Trainer:
    def __init__(self, model, dataset, ckpt_mng:tf.train.CheckpointManager,
                        val_evaluator:Evaluator = None,
                        optimizer = keras.optimizers.SGD(),
                        config:Config = Config(),
                        augmentation = None,
                        logs_dir='logs/'):
        self.config = config
        self.val_evaluator = val_evaluator
        self.ckpt_mng = ckpt_mng

        self.summary_writer = tf.summary.create_file_writer(logs_dir)
        dataset.prepare()
        
        self.mirrored_strategy = config.MIRRORED_STRATEGY

        with self.mirrored_strategy.scope():
            self.optimizer = optimizer
            self.model = model
            self.dataset = tf.data.Dataset.from_generator(lambda : data_generator(dataset, config, augmentation=augmentation, batch_size=1),
                                                            output_types=(((tf.float32, tf.float64, tf.int32, tf.float64, tf.int32, tf.int32, tf.bool),())))\
                                            .batch(config.BATCH_SIZE)\
                                            .map(lambda inputs,_: [tf.squeeze(t,axis=1) for t in inputs],num_parallel_calls=tf.data.AUTOTUNE)\
                                            .prefetch(tf.data.AUTOTUNE)
            self.dataset = self.mirrored_strategy.experimental_distribute_dataset(self.dataset)

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

        for epoch in range(max_epoch):
            pbar = tqdm(desc=f'Epoch : {epoch+1}/{max_epoch}',unit='step', total = self.config.STEPS_PER_EPOCH)
            with self.mirrored_strategy.scope():
                for i, inputs in enumerate(self.dataset):
                    if self.config.STEPS_PER_EPOCH > i:
                        mean_losses = self.train_step(inputs)
                        (rpn_bbox_loss, rpn_class_loss, class_loss, bbox_loss, mask_loss), reg_losses, total_loss= mean_losses

                        with self.summary_writer.as_default():
                            tf.summary.scalar('rpn_class_loss', rpn_class_loss, step=self.optimizer.iterations)
                            tf.summary.scalar('rpn_bbox_loss', rpn_bbox_loss, step=self.optimizer.iterations)
                            tf.summary.scalar('mrcnn_class_loss', class_loss, step=self.optimizer.iterations)
                            tf.summary.scalar('mrcnn_bbox_loss', bbox_loss, step=self.optimizer.iterations)
                            tf.summary.scalar('mrcnn_mask_loss', mask_loss, step=self.optimizer.iterations)
                            tf.summary.scalar('reg_losses', reg_losses, step=self.optimizer.iterations)
                            tf.summary.scalar('total_loss', total_loss, step=self.optimizer.iterations)
                    else:
                        break

                    pbar.update()
                    pbar.set_postfix({'total_loss':total_loss.numpy(),
                                       'lr': self.optimizer._decayed_lr('float32').numpy()})
            pbar.close()
            self.ckpt_mng.save()

            if self.val_evaluator is not None:
                val_metric = self.val_evaluator.eval(limit_step=self.config.VALIDATION_STEPS)
                with self.summary_writer.as_default():
                    tf.summary.scalar('val_mAP', val_metric['mAP'], step=self.optimizer.iterations)
                    tf.summary.scalar('val_recall', val_metric['recall'], step=self.optimizer.iterations)
                    tf.summary.scalar('val_precision', val_metric['precision'], step=self.optimizer.iterations)
                    tf.summary.scalar('val_F1-Score', val_metric['F1-Score'], step=self.optimizer.iterations)


    @tf.function
    def train_step(self, dist_inputs):
        def step_fn(inputs):
            images, input_image_meta, \
            input_rpn_match, input_rpn_bbox, \
            input_gt_class_ids, input_gt_boxes, input_gt_masks = inputs
            with tf.GradientTape() as tape:
                outputs = self.model(images, input_image_meta,
                                    input_rpn_match=input_rpn_match, 
                                    input_rpn_bbox=input_rpn_bbox, 
                                    input_gt_class_ids=input_gt_class_ids, 
                                    input_gt_boxes=input_gt_boxes, 
                                    input_gt_masks=input_gt_masks, 
                                    input_rois = None,
                                    training=True)
                reg_losses, total_loss = self.cal_loss(outputs)
            
            grads = tape.gradient(total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(list(zip(grads, self.model.trainable_variables)))
            return outputs, reg_losses, total_loss
        
        per_example_losses = self.mirrored_strategy.run(step_fn, args=(dist_inputs,))
        mean_losses = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_example_losses,axis=None)
        return mean_losses
    
    @tf.function
    def cal_loss(self, outputs):
        rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss = outputs
        reg_losses = tf.add_n([keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
                                            for w in self.model.trainable_weights
                                            if 'gamma' not in w.name and 'beta' not in w.name])

        total_loss = tf.reduce_sum([rpn_bbox_loss, rpn_class_loss, class_loss, bbox_loss, mask_loss, reg_losses])

        return reg_losses, total_loss