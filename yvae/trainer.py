from models import *
import numpy as np
import tensorflow as tf

class YVAE_Trainer():
    def __init__(self,y_vae_list,epochs,dataset_dict,optimizer,callbacks=[]):
        assert len(y_vae_list)==len(dataset_dict)
        self.y_vae_list=y_vae_list
        self.epochs=epochs
        self.dataset_names=[k for k in dataset_dict.keys()]
        self.dataset_list=[v for v in dataset_dict.values()]
        self.optimizer=optimizer
        self.callbacks=callbacks

  #@tf.function
    def train_step(self,batch,vae):
        with tf.GradientTape() as tape:
            [reconstruction,z_mean, z_log_var]=vae(batch)
            reconstruction_loss = tf.reduce_mean(
                        tf.reduce_sum(
                            keras.losses.binary_crossentropy(batch, reconstruction), axis=(1, 2)
                        )
                    )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, vae.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, vae.trainable_weights))
        return total_loss

    def train_loop(self):
        for e in range(self.epochs):
            epoch_losses=[0 for _ in self.dataset_list]
            for d,dataset in enumerate(self.dataset_list):
                vae=self.y_vae_list[d]
                for batch in dataset:
                    total_loss=self.train_step(batch,vae)
                    epoch_losses[d]+=total_loss
            print([e.numpy() for e in epoch_losses])
            print('epoch {} sum: {} mean: {} std dev: {}'.format(e,np.sum(epoch_losses), np.mean(epoch_losses), np.std(epoch_losses)))
            for callback in self.callbacks:
                callback(self,e)
        return np.mean(epoch_losses)
