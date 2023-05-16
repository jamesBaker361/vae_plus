from yvae_model import *
import numpy as np
import tensorflow as tf
import time

class YVAE_Classifier_Trainer:
    def __init__(self,classifier_model,epochs,optimizer,dataset,start_epoch=0,callbacks=[]):
        self.classifier_model=classifier_model
        self.encoder=classifier_model.get_layer('encoder')
        self.classification_head=classifier_model.get_layer('classification_head')
        self.epochs=epochs
        self.start_epoch=start_epoch
        self.optimizer=optimizer
        self.dataset=dataset
        self.loss_function=tf.keras.losses.CategoricalCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.SUM)
        self.callbacks=callbacks

    @tf.function
    def train_step(self,batch):
        with tf.GradientTape() as tape:
            (imgs,labels)=batch
            predictions=self.classifier_model(imgs)
            loss=self.loss_function(predictions,labels)
        grads = tape.gradient(loss, self.classifier_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.classifier_model.trainable_weights))
        return loss
    
    def train_loop(self):
        epoch_losses=[]
        for e in range(self.start_epoch,self.epochs):
            start = time.time()
            batch_losses=[]
            for batch in self.dataset:
                loss=self.train_step(batch)
                batch_losses.append(loss)
            sum_loss=np.sum(batch_losses)
            print('epoch {} sum: {} std: {} mean: {}'.format(e, sum_loss, np.std(batch_losses), np.mean(batch_losses)))
            print ('\nTime taken for epoch {} is {} sec\n'.format(e,time.time()-start))
            epoch_losses.append(sum_loss)
            for callback in self.callbacks:
                callback(e)
        return sum_loss