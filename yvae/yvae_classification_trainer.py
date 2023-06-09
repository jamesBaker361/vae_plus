from yvae_model import *
from processing_utils import *
import numpy as np
import tensorflow as tf
import time

TEST_INTERVAL=1
EPSILON= 1e-8
import random
TRAIN_LOSS='train_loss'
TEST_LOSS='test_loss'

class YVAE_Classifier_Trainer:
    def __init__(self,classifier_model,epochs,optimizer,dataset,test_dataset,log_dir='',mirrored_strategy=None,start_epoch=0,callbacks=[],use_external=False,unfreezing_epoch=-1,unfrozen_optimizer=None, data_augmentation=False):
        self.classifier_model=classifier_model
        self.epochs=epochs
        self.start_epoch=start_epoch
        self.optimizer=optimizer
        self.dataset=dataset
        self.test_dataset=test_dataset
        self.callbacks=callbacks
        self.mirrored_strategy=mirrored_strategy
        self.log_dir=log_dir
        self.summary_writer = tf.summary.create_file_writer(log_dir)
        self.use_external=use_external #if we're using an external pretrained model like mobilenet or efficientnet
        self.unfreezing_epoch=unfreezing_epoch
        self.unfrozen_optimizer=unfrozen_optimizer
        self.data_augmentation=data_augmentation
        if self.mirrored_strategy is not None:
            with self.mirrored_strategy.scope():
                self.loss_function=tf.keras.losses.CategoricalCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)
                self.train_loss = tf.keras.metrics.Mean(TRAIN_LOSS, dtype=tf.float32)
                self.test_loss= tf.keras.metrics.Mean(TEST_LOSS, dtype=tf.float32)
                self.data_augmenter= tf.keras.Sequential(
                    [tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"), tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),]
                    )
        else:
            self.loss_function=tf.keras.losses.CategoricalCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)
            self.train_loss = tf.keras.metrics.Mean(TRAIN_LOSS, dtype=tf.float32)
            self.test_loss= tf.keras.metrics.Mean(TEST_LOSS, dtype=tf.float32)
            self.data_augmenter= tf.keras.Sequential(
                    [tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"), tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),]
                    )

    def train_step(self,batch):
        with tf.GradientTape() as tape:
            (imgs,labels)=batch
            if self.data_augmentation:
                imgs=self.data_augmenter(imgs)
            predictions=self.classifier_model(imgs) + EPSILON
            loss=self.loss_function(labels, predictions)
        self.train_loss(loss)
        grads = tape.gradient(loss, self.classifier_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.classifier_model.trainable_weights))
        return loss

    def test_step(self,batch):
        (imgs,labels)=batch
        predictions=self.classifier_model(imgs)
        loss=self.loss_function(labels, predictions)
        if random.randint(0,30) % 13==0:
            print('unlucky number :(')
            print('label', labels)
            print('predictions', predictions)
            print('loss', loss)
        self.test_loss(loss)
        return loss
    
    @tf.function
    def distributed_train_step(self,batch):
        per_replica_losses = self.mirrored_strategy.run(self.train_step, args=(batch,))
        return self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                axis=None)
    
    @tf.function
    def distributed_test_step(self,batch):
        per_replica_losses = self.mirrored_strategy.run(self.test_step, args=(batch,))
        return self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                axis=None)

    def train_loop(self):
        for e in range(self.start_epoch,self.epochs):
            #self.reset_metrics()
            if self.use_external and  self.unfreezing_epoch<=e:
                self.optimizer=self.unfrozen_optimizer
                self.classifier_model.get_layer(EXTERNAL_MODEL).trainable=True
            start = time.time()
            batch_losses=[]
            for batch in self.dataset:
                if self.mirrored_strategy is None:
                    loss=self.train_step(batch)
                else:
                    loss=self.distributed_train_step(batch)
                batch_losses.append(loss)
            print('epoch {} loss: {}'.format(e, self.train_loss.result()))
            print ('\nTime taken for epoch {} is {} sec\n'.format(e,time.time()-start))
            with self.summary_writer.as_default():
                tf.summary.scalar(TRAIN_LOSS, self.train_loss.result(), step=e)
            for callback in self.callbacks:
                callback(e)
            (imgs,labels)=next(iter(self.test_dataset))
            predictions=self.classifier_model(imgs) + EPSILON
            loss=self.loss_function(labels,predictions)
            print('loss', loss)
            print('predictions:',predictions)
            print('labels', labels)
            print('shape', tf.shape(imgs))
            loss=self.loss_function(labels,predictions)
            print('loss', loss)
            if e%TEST_INTERVAL==0:
                start = time.time()
                for batch in self.test_dataset:
                    if self.mirrored_strategy is None:
                        loss=self.test_step(batch)
                    else:
                        loss=self.distributed_test_step(batch)
                print('epoch {} test loss: {}'.format(e, self.test_loss.result()))
                print ('\nTime taken for test epoch {} is {} sec\n'.format(e,time.time()-start))
                with self.summary_writer.as_default():
                    tf.summary.scalar(TEST_LOSS, self.test_loss.result(), step=e)
        for callback in self.callbacks:
            callback(e)

