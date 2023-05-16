from yvae_model import *
import numpy as np
import tensorflow as tf
import time


class VAE_Trainer:
    def __init__(self,vae_list,epochs,dataset_dict,optimizer,kl_loss_scale,callbacks,start_epoch):
        self.vae_list=vae_list
        self.epochs=epochs
        self.dataset_names=[k for k in dataset_dict.keys()]
        self.dataset_list=[v for v in dataset_dict.values()]
        self.optimizer=optimizer
        self.callbacks=callbacks
        self.start_epoch=start_epoch
        self.kl_loss_scale=kl_loss_scale
        self.reconstruction_loss_function=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)

    def train_step(self,batch,vae):
        with tf.GradientTape() as tape:
            [reconstruction,z_mean, z_log_var]=vae(batch)
            reconstruction_loss =self.reconstruction_loss_function(batch, reconstruction)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = self.kl_loss_scale * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, vae.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, vae.trainable_weights))
        return total_loss

    def train_loop(self):
        for e in range(self.start_epoch,self.epochs):
            start = time.time()
            epoch_losses=[0 for _ in self.dataset_list]
            for d,dataset in enumerate(self.dataset_list):
                vae=self.vae_list[d]
                for batch in dataset:
                    total_loss=self.train_step(batch,vae)
                    epoch_losses[d]+=total_loss
            #print([ep.numpy() for ep in epoch_losses])
            print('epoch {} sum: {} mean: {} std dev: {}'.format(e,np.sum(epoch_losses), np.mean(epoch_losses), np.std(epoch_losses)))
            print ('\nTime taken for epoch {} is {} sec\n'.format(e,time.time()-start))
            for callback in self.callbacks:
                callback(e)
        return np.mean(epoch_losses)

class YVAE_Trainer(VAE_Trainer):
    def __init__(self,y_vae_list,epochs,dataset_dict,optimizer,reconstruction_loss_function_name,kl_loss_scale=1.0,callbacks=[],start_epoch=0):
        super().__init__(y_vae_list,epochs,dataset_dict,optimizer,kl_loss_scale,callbacks,start_epoch)
        self.decoders=[y_vae_list[i].get_layer('decoder_{}'.format(i)) for i in range(len(y_vae_list))]
        self.encoder=y_vae_list[0].get_layer('encoder')
        if reconstruction_loss_function_name == 'binary_crossentropy':
            self.reconstruction_loss_function=tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
        elif reconstruction_loss_function_name == 'mse':
            self.reconstruction_loss_function=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
        elif reconstruction_loss_function_name == 'log_cosh':
            self.reconstruction_loss_function=tf.keras.losses.LogCosh(reduction=tf.keras.losses.Reduction.SUM)
        elif reconstruction_loss_function_name == 'huber':
            self.reconstruction_loss_function=tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    
    def generate_images(self,batch_size):
        noise_shape=self.decoders[0].input_shape[1:]
        print(noise_shape)
        noise=tf.random.normal((batch_size, *noise_shape))
        return [decoder(noise) for decoder in self.decoders]
