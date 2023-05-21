from yvae_model import *
import numpy as np
import tensorflow as tf
import time

TRAIN='/train'
TEST='/test'
TEST_INTERVAL=10

class VAE_Trainer:
    def __init__(self,vae_list,epochs,dataset_dict,test_dataset_dict,optimizer,log_dir='',mirrored_strategy=None,kl_loss_scale=1.0,callbacks=[],start_epoch=0):
        self.vae_list=vae_list
        self.decoders=[vae_list[i].get_layer('decoder_{}'.format(i)) for i in range(len(vae_list))]
        self.epochs=epochs
        self.dataset_names=[k for k in dataset_dict.keys()]
        self.dataset_list=[v for v in dataset_dict.values()]
        self.test_dataset_list=[v for v in test_dataset_dict.values()]
        self.optimizer=optimizer
        self.callbacks=callbacks
        self.start_epoch=start_epoch
        self.kl_loss_scale=kl_loss_scale
        self.reconstruction_loss_function=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.log_dir=log_dir
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.train_recontruction_loss= tf.keras.metrics.Mean('train_reconstruction_loss', dtype=tf.float32)
        self.test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
        self.test_recontruction_loss= tf.keras.metrics.Mean('test_reconstruction_loss', dtype=tf.float32)
        self.summary_writer = tf.summary.create_file_writer(log_dir)
        self.mirrored_strategy=mirrored_strategy

    def train_step(self,batch,vae):
        with tf.GradientTape() as tape:
            [reconstruction,z_mean, z_log_var]=vae(batch)
            reconstruction_loss =self.reconstruction_loss_function(batch, reconstruction)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = self.kl_loss_scale * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, vae.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, vae.trainable_weights))
        self.train_loss(total_loss)
        self.train_recontruction_loss(reconstruction_loss)
        return total_loss, reconstruction_loss
    
    @tf.function
    def distributed_train_step(self,batch,vae):
        per_replica_losses = self.mirrored_strategy.run(self.train_step, args=(batch,vae,))
        return self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                axis=None)
    
    def test_step(self,batch,vae):
        [reconstruction,z_mean, z_log_var]=vae(batch)
        reconstruction_loss =self.reconstruction_loss_function(batch, reconstruction)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = self.kl_loss_scale * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss
        return total_loss, reconstruction_loss

    def train_loop(self):
        for e in range(self.start_epoch,self.epochs):
            start = time.time()
            epoch_losses=[0 for _ in self.dataset_list]
            for d,dataset in enumerate(self.dataset_list):
                vae=self.vae_list[d]
                for batch in dataset:
                    if self.mirrored_strategy is None:
                        total_loss,reconstruction_loss=self.train_step(batch,vae)
                    else:
                        total_loss,reconstruction_loss=self.distributed_train_step(batch,vae)
                    
            #print([ep.numpy() for ep in epoch_losses])
            print('epoch {} loss: {}'.format(e,self.train_loss.result()))
            print ('\nTime taken for epoch {} is {} sec\n'.format(e,time.time()-start))
            with self.summary_writer.as_default():
                tf.summary.scalar('train_loss', self.train_loss.result(), step=e)
                tf.summary.scalar('train_reconstruction_loss', self.train_recontruction_loss.result(), step=e)
            for callback in self.callbacks:
                callback(e)
            if e%TEST_INTERVAL==0:
                start = time.time()
                for d,dataset in enumerate(self.test_dataset_list):
                    vae=self.vae_list[d]
                    for batch in dataset:
                        total_loss,reconstruction_loss=self.test_step(batch,vae)
                        self.test_loss(total_loss)
                        self.test_recontruction_loss(reconstruction_loss)
                print('\ntest epoch {} mean: {} '.format(e,self.test_loss.result()))
                print ('\nTime taken for test epoch {} is {} sec\n'.format(e,time.time()-start))
                with self.summary_writer.as_default():
                    tf.summary.scalar('test_loss', self.test_loss.result(), step=e)
                    tf.summary.scalar('test_reconstruction_loss', self.test_recontruction_loss.result(), step=e)
    
    
    def generate_images(self,batch_size):
        noise_shape=self.decoders[0].input_shape[1:]
        print(noise_shape)
        noise=tf.random.normal((batch_size, *noise_shape))
        return [decoder(noise) for decoder in self.decoders]

class YVAE_Trainer(VAE_Trainer):
    def __init__(self,y_vae_list,epochs,dataset_dict, test_dataset_dict,optimizer,reconstruction_loss_function_name,log_dir='', mirrored_strategy=None,kl_loss_scale=1.0,callbacks=[],start_epoch=0):
        super().__init__(y_vae_list,epochs,dataset_dict,test_dataset_dict,optimizer,log_dir=log_dir,mirrored_strategy=mirrored_strategy ,kl_loss_scale=kl_loss_scale,callbacks=callbacks,start_epoch=start_epoch)
        self.encoder=y_vae_list[0].get_layer('encoder')
        if reconstruction_loss_function_name == 'binary_crossentropy':
            self.reconstruction_loss_function=tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        elif reconstruction_loss_function_name == 'mse':
            self.reconstruction_loss_function=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        elif reconstruction_loss_function_name == 'log_cosh':
            self.reconstruction_loss_function=tf.keras.losses.LogCosh(reduction=tf.keras.losses.Reduction.NONE)
        elif reconstruction_loss_function_name == 'huber':
            self.reconstruction_loss_function=tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
