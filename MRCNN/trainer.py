import os
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pickle as pk

from tqdm import tqdm

from .evaluator import Evaluator
from MRCNN.config import Config
from MRCNN.data.data_generator import data_generator


class Trainer:
    def __init__(self, model, dataset,
                        val_evaluator:Evaluator = None,
                        optimizer = keras.optimizers.SGD(),
                        config:Config = Config(),
                        augmentation = None,
                        logs_dir='logs/'):
        self.config = config
        self.val_evaluator = val_evaluator

        self.summary_writer = tf.summary.create_file_writer(logs_dir)
        dataset.prepare()
        
        self.mirrored_strategy = config.STRATEGY
        self.model = model
        # for i in data_generator(dataset, config, augmentation=augmentation, batch_size=1):
        #     print(i)

        with self.mirrored_strategy.scope():
            self.optimizer = optimizer
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


        ckpt = tf.train.Checkpoint(model=self.model)
        ckpt_mng = tf.train.CheckpointManager(ckpt, directory='save_model', max_to_keep=None)


        self.model.set_trainable(layers)

        for epoch in range(max_epoch):
            cls_attentions_sum = np.zeros([self.config.STEPS_PER_EPOCH, self.config.NUM_CLASSES, 2048])
            cls_attentions_cnt = np.zeros([self.config.STEPS_PER_EPOCH, self.config.NUM_CLASSES, 1])
            pbar = tqdm(desc=f'Epoch : {epoch+1}/{max_epoch}',unit='step', total = self.config.STEPS_PER_EPOCH)
            with self.mirrored_strategy.scope():
                for i, inputs in enumerate(self.dataset):
                    if self.config.STEPS_PER_EPOCH > i:
                        losses, attentions, prn_cls = self.train_step(inputs)

                        rpn_bbox_loss, rpn_class_loss, class_loss, bbox_loss, mask_loss, reg_losses, meta_loss= losses

                        mean_loss = np.nanmean([loss.numpy() for loss in losses])

                        for cls_idx, prn_cls_id in enumerate(prn_cls):
                            cls_attentions_sum[i,prn_cls_id,...] += attentions[cls_idx]
                            cls_attentions_cnt[i,prn_cls_id,...] += 1

                        with self.summary_writer.as_default():
                            tf.summary.scalar('rpn_class_loss', rpn_class_loss, step=self.optimizer.iterations)
                            tf.summary.scalar('rpn_bbox_loss', rpn_bbox_loss, step=self.optimizer.iterations)
                            tf.summary.scalar('mrcnn_class_loss', class_loss, step=self.optimizer.iterations)
                            tf.summary.scalar('mrcnn_bbox_loss', bbox_loss, step=self.optimizer.iterations)
                            tf.summary.scalar('mrcnn_mask_loss', mask_loss, step=self.optimizer.iterations)
                            tf.summary.scalar('reg_losses', reg_losses, step=self.optimizer.iterations)
                            tf.summary.scalar('meta_loss', meta_loss, step=self.optimizer.iterations)
                    else:
                        break

                    pbar.update()
                    pbar.set_postfix({'mean_loss':mean_loss,
                                       'lr': self.optimizer._decayed_lr('float32').numpy()})

            attentions = cls_attentions_sum/cls_attentions_cnt
            attentions = np.where(attentions==0, np.nan, attentions)
            attentions = np.nanmean(attentions,0)
            if not os.path.isdir('save_attentions'):
                os.mkdir('save_attentions')
            with open(f'save_attentions/{epoch}.pickle', 'wb') as f:
                pk.dump(attentions, f)
            pbar.close()
            ckpt_mng.save()

            if self.val_evaluator is not None:
                val_metric = self.val_evaluator.eval(attentions, limit_step=self.config.VALIDATION_STEPS, iouType='bbox', per_class=False)
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
            
            grads = tape.gradient(self.model.losses, self.model.trainable_variables)
            self.optimizer.apply_gradients(list(zip(grads, self.model.trainable_variables)))
            return outputs
        
        losses, attentions = self.mirrored_strategy.run(step_fn, args=(dist_inputs,))
        mean_losses = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, losses,axis=None)
        attentions, prn_cls = attentions
        attentions = self.mirrored_strategy.gather(attentions,axis=0)
        prn_cls = self.mirrored_strategy.gather(prn_cls,axis=0)
        return mean_losses, attentions, prn_cls