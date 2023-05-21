from yvae_model import *
import numpy as np
import tensorflow as tf
import time

class YVAE_Classifier_Trainer:
    def __init__(self,classifier_model,epochs,optimizer,dataset,log_dir='',mirrored_strategy=None,start_epoch=0,callbacks=[]):
        self.classifier_model=classifier_model
        self.encoder=classifier_model.get_layer('encoder')
        self.classification_head=classifier_model.get_layer('classification_head')
        self.epochs=epochs
        self.start_epoch=start_epoch
        self.optimizer=optimizer
        self.dataset=dataset
        self.loss_function=tf.keras.losses.CategoricalCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)
        self.callbacks=callbacks
        self.mirrored_strategy=mirrored_strategy
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.log_dir=log_dir
        self.summary_writer = tf.summary.create_file_writer(log_dir)

    
    def train_step(self,batch):
        with tf.GradientTape() as tape:
            (imgs,labels)=batch
            predictions=self.classifier_model(imgs)
            loss=self.loss_function(predictions,labels)
        self.train_loss(loss)
        grads = tape.gradient(loss, self.classifier_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.classifier_model.trainable_weights))
        return loss
    
    @tf.function
    def distributed_train_step(self,batch):
        per_replica_losses = self.mirrored_strategy.run(self.train_step, args=(batch,))
        return self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                axis=None)
    
    def train_loop(self):
        for e in range(self.start_epoch,self.epochs):
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
                tf.summary.scalar('train_loss', self.train_loss.result(), step=e)
            for callback in self.callbacks:
                callback(e)