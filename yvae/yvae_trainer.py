from yvae_model import *
import numpy as np
import tensorflow as tf
import time

TRAIN='/train'
TEST='/test'
TEST_INTERVAL=10
TRAIN_LOSS='train_loss'
TEST_LOSS='test_loss'
TRAIN_RECONSTRUCTION_LOSS='train_reconstruction_loss'
TEST_RECONSTRUCTION_LOSS='test_reconstruction_loss'
TRAIN_CREATIVITY_LOSS='train_creativity_loss'
TEST_CREATIVITY_LOSS='test_creativity_loss'

def get_compute_kl_loss(kl_loss_scale, global_batch_size):
    def _compute_kl_loss(z_mean, z_log_var):
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = kl_loss_scale * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        per_example_loss = [kl_loss]
        return tf.nn.compute_average_loss(per_example_loss,
                                      global_batch_size=global_batch_size)
    
    return _compute_kl_loss

def get_compute_creativity_loss(reconstruction_loss_function, creativity_lambda, global_batch_size,n_classes):
    def _compute_creativity_loss(predicted_labels):
        fill_value=1.0/n_classes
        batch_size=tf.shape(predicted_labels)[0]
        desired_output= tf.fill(dims=(batch_size, n_classes), value=fill_value)
        per_example_loss= creativity_lambda* reconstruction_loss_function(desired_output, predicted_labels)
        return tf.nn.compute_average_loss(per_example_loss,
                                      global_batch_size=global_batch_size)
    return _compute_creativity_loss

class VAE_Trainer:
    def __init__(self,vae_list,epochs,dataset_dict,test_dataset_dict,optimizer,log_dir='',mirrored_strategy=None,kl_loss_scale=1.0,callbacks=[],start_epoch=0,global_batch_size=4):
        self.vae_list=vae_list
        self.decoders=[vae_list[i].get_layer(DECODER_NAME.format(i)) for i in range(len(vae_list))]
        self.epochs=epochs
        self.dataset_names=[k for k in dataset_dict.keys()]
        self.dataset_list=[v for v in dataset_dict.values()]
        self.test_dataset_list=[v for v in test_dataset_dict.values()]
        self.optimizer=optimizer
        self.callbacks=callbacks
        self.start_epoch=start_epoch
        self.kl_loss_scale=kl_loss_scale
        self.log_dir=log_dir
        
        self.mirrored_strategy=mirrored_strategy
        if mirrored_strategy is not None:
            with mirrored_strategy.scope():
                self.reconstruction_loss_function=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
                self.train_loss = tf.keras.metrics.Mean(TRAIN_LOSS, dtype=tf.float32)
                self.train_reconstruction_loss= tf.keras.metrics.Mean(TRAIN_RECONSTRUCTION_LOSS, dtype=tf.float32)
                self.test_loss = tf.keras.metrics.Mean(TEST_LOSS, dtype=tf.float32)
                self.test_reconstruction_loss= tf.keras.metrics.Mean(TEST_RECONSTRUCTION_LOSS, dtype=tf.float32)
                self.summary_writer = tf.summary.create_file_writer(log_dir)
                self.compute_kl_loss=get_compute_kl_loss(kl_loss_scale, global_batch_size)
        else:
            self.reconstruction_loss_function=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
            self.train_loss = tf.keras.metrics.Mean(TRAIN_LOSS, dtype=tf.float32)
            self.train_reconstruction_loss= tf.keras.metrics.Mean(TRAIN_RECONSTRUCTION_LOSS, dtype=tf.float32)
            self.test_loss = tf.keras.metrics.Mean(TEST_LOSS, dtype=tf.float32)
            self.test_reconstruction_loss= tf.keras.metrics.Mean(TEST_RECONSTRUCTION_LOSS, dtype=tf.float32)
            self.summary_writer = tf.summary.create_file_writer(log_dir)
            self.compute_kl_loss=get_compute_kl_loss(kl_loss_scale, global_batch_size)
        self.train_metrics={
            TRAIN_LOSS:self.train_loss,
            TRAIN_RECONSTRUCTION_LOSS:self.train_reconstruction_loss
        }
        self.test_metrics={
            TEST_LOSS:self.test_loss,
            TEST_RECONSTRUCTION_LOSS: self.test_reconstruction_loss
        }

    def train_step(self,batch,vae):
        with tf.GradientTape() as tape:
            [reconstruction,z_mean, z_log_var]=vae(batch)
            reconstruction_loss =self.reconstruction_loss_function(batch, reconstruction) #yes this is redundant whoops
            total_loss=self.compute_kl_loss(z_mean,z_log_var) +reconstruction_loss
        grads = tape.gradient(total_loss, vae.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, vae.trainable_weights))
        self.train_loss(total_loss)
        self.train_reconstruction_loss(reconstruction_loss)
        return total_loss
    
    @tf.function
    def distributed_train_step(self,batch,vae):
        per_replica_losses = self.mirrored_strategy.run(self.train_step, args=(batch,vae,))
        return self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                axis=None)
    
    @tf.function
    def distributed_test_step(self,batch,vae):
        per_replica_losses = self.mirrored_strategy.run(self.test_step, args=(batch,vae,))
        return self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                axis=None)
    
    def test_step(self,batch,vae):
        [reconstruction,z_mean, z_log_var]=vae(batch)
        reconstruction_loss =self.reconstruction_loss_function(batch, reconstruction)
        total_loss=self.compute_kl_loss(z_mean,z_log_var)
        self.test_loss(total_loss)
        self.test_reconstruction_loss(reconstruction_loss)
        return total_loss

    def train_loop(self):
        for e in range(self.start_epoch,self.epochs):
            start = time.time()
            epoch_losses=[0 for _ in self.dataset_list]
            for d,dataset in enumerate(self.dataset_list):
                vae=self.vae_list[d]
                for batch in dataset:
                    if self.mirrored_strategy is None:
                        total_loss=self.train_step(batch,vae)
                    else:
                        total_loss=self.distributed_train_step(batch,vae)
                    
            #print([ep.numpy() for ep in epoch_losses])
            print('epoch {} loss: {}'.format(e,self.train_loss.result()))
            print ('\nTime taken for epoch {} is {} sec\n'.format(e,time.time()-start))
            with self.summary_writer.as_default():
                for name,metric in self.train_metrics.items():
                    tf.summary.scalar(name, metric.result(), step=e)
            for callback in self.callbacks:
                callback(e)
            if e%TEST_INTERVAL==0:
                start = time.time()
                for d,dataset in enumerate(self.test_dataset_list):
                    vae=self.vae_list[d]
                    for batch in dataset:
                        if self.mirrored_strategy is None:
                            total_loss=self.test_step(batch,vae)
                        else:
                            total_loss=self.distributed_test_step(batch,vae)
                print('\ntest epoch {} mean: {} '.format(e,self.test_loss.result()))
                print ('\nTime taken for test epoch {} is {} sec\n'.format(e,time.time()-start))
                with self.summary_writer.as_default():
                    for name,metric in self.test_metrics.items():
                        tf.summary.scalar(name, metric.result(), step=e)
    
    
    def generate_images(self,batch_size):
        noise_shape=self.decoders[0].input_shape[1:]
        print(noise_shape)
        noise=tf.random.normal((batch_size, *noise_shape))
        return [decoder(noise) for decoder in self.decoders]

class VAE_Unit_Trainer(VAE_Trainer):
    def __init__(self,vae_list,epochs,dataset_dict,test_dataset_dict,optimizer,log_dir='',mirrored_strategy=None,kl_loss_scale=1.0,callbacks=[],start_epoch=0):
        super().__init__(vae_list,epochs,dataset_dict,test_dataset_dict,optimizer,log_dir=log_dir,mirrored_strategy=mirrored_strategy ,kl_loss_scale=kl_loss_scale,callbacks=callbacks,start_epoch=start_epoch)
        vae_list[0].summary()
        self.shared_partial=vae_list[0].get_layer(ENCODER_STEM_NAME.format(0)).get_layer(SHARED_ENCODER_NAME)
        self.partials=[vae_list[i].get_layer(ENCODER_STEM_NAME.format(i)).get_layer(PARTIAL_ENCODER_NAME.format(i)) for i in range(len(vae_list))]


class VAE_Creativity_Trainer(VAE_Trainer):
    def __init__(self,vae_list,epochs,dataset_dict,test_dataset_dict,optimizer,dataset_list,log_dir='',mirrored_strategy=None,kl_loss_scale=1.0,callbacks=[],start_epoch=0, global_batch_size=4,pretrained_classifier=None, creativity_lambda=1.0,n_classes=2):
        super().__init__(vae_list,epochs,dataset_dict={},test_dataset_dict={},optimizer=optimizer,log_dir=log_dir,mirrored_strategy=mirrored_strategy ,kl_loss_scale=kl_loss_scale,callbacks=callbacks,start_epoch=start_epoch,global_batch_size=global_batch_size)
        self.dataset_list=[dataset_list] #should be a list of images and shit from all classes
        self.creativity_lambda=creativity_lambda
        self.pretrained_classifier=pretrained_classifier
        self.n_classes=n_classes
        if mirrored_strategy is not None:
            self.train_creativity_loss=tf.keras.metrics.Mean(TRAIN_CREATIVITY_LOSS, dtype=tf.float32)
            self.test_creativity_loss=tf.keras.metrics.Mean(TEST_CREATIVITY_LOSS, dtype=tf.float32)
            self.reconstruction_loss_function=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
            self.compute_creativity_loss=get_compute_creativity_loss(self.reconstruction_loss_function, creativity_lambda, global_batch_size,n_classes)
            self.train_metrics[TRAIN_CREATIVITY_LOSS]=self.train_creativity_loss
            self.test_metrics[TEST_CREATIVITY_LOSS]=self.test_creativity_loss
        else:
            self.train_creativity_loss=tf.keras.metrics.Mean(TRAIN_CREATIVITY_LOSS, dtype=tf.float32)
            self.test_creativity_loss=tf.keras.metrics.Mean(TEST_CREATIVITY_LOSS, dtype=tf.float32)
            self.reconstruction_loss_function=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
            self.compute_creativity_loss=get_compute_creativity_loss(self.reconstruction_loss_function, creativity_lambda, global_batch_size,n_classes)
            self.train_metrics[TRAIN_CREATIVITY_LOSS]=self.train_creativity_loss
            self.test_metrics[TEST_CREATIVITY_LOSS]=self.test_creativity_loss

    def train_step(self,batch,vae):
        with tf.GradientTape() as tape:
            [reconstruction,z_mean, z_log_var]=vae(batch)
            reconstruction_loss =self.reconstruction_loss_function(batch, reconstruction)
            kl_loss=self.compute_kl_loss(z_mean,z_log_var)
            predicted_labels=self.pretrained_classifier(reconstruction)
            creativity_loss=self.compute_creativity_loss(predicted_labels)
            total_loss= reconstruction_loss+kl_loss+creativity_loss
        grads = tape.gradient(total_loss, vae.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, vae.trainable_weights))
        self.train_loss(total_loss)
        self.train_reconstruction_loss(reconstruction_loss)
        self.train_creativity_loss(creativity_loss)
        return total_loss
    
    def test_step(self,batch,vae):
        with tf.GradientTape() as tape:
            [reconstruction,z_mean, z_log_var]=vae(batch)
            reconstruction_loss =self.reconstruction_loss_function(batch, reconstruction)
            kl_loss=self.compute_kl_loss(z_mean,z_log_var)
            predicted_labels=self.pretrained_classifier(batch)
            creativity_loss=self.compute_creativity_loss(predicted_labels)
            total_loss= reconstruction_loss+kl_loss+creativity_loss
        grads = tape.gradient(total_loss, vae.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, vae.trainable_weights))
        self.train_loss(total_loss)
        self.train_reconstruction_loss(reconstruction_loss)
        self.train_creativity_loss(creativity_loss)
        return total_loss
    

class YVAE_Trainer(VAE_Trainer):
    def __init__(self,y_vae_list,epochs,dataset_dict, test_dataset_dict,optimizer,reconstruction_loss_function_name='mse',log_dir='', mirrored_strategy=None,kl_loss_scale=1.0,callbacks=[],start_epoch=0,global_batch_size=4):
        super().__init__(y_vae_list,epochs,dataset_dict,test_dataset_dict,optimizer,log_dir=log_dir,mirrored_strategy=mirrored_strategy ,kl_loss_scale=kl_loss_scale,callbacks=callbacks,start_epoch=start_epoch,global_batch_size=global_batch_size)
        self.encoder=y_vae_list[0].get_layer('encoder')
        if mirrored_strategy is not None:
            with mirrored_strategy.scope():
                if reconstruction_loss_function_name == 'binary_crossentropy':
                    self.reconstruction_loss_function=tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
                elif reconstruction_loss_function_name == 'mse':
                    self.reconstruction_loss_function=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
                elif reconstruction_loss_function_name == 'log_cosh':
                    self.reconstruction_loss_function=tf.keras.losses.LogCosh(reduction=tf.keras.losses.Reduction.NONE)
                elif reconstruction_loss_function_name == 'huber':
                    self.reconstruction_loss_function=tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
        else:
            if reconstruction_loss_function_name == 'binary_crossentropy':
                self.reconstruction_loss_function=tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
            elif reconstruction_loss_function_name == 'mse':
                self.reconstruction_loss_function=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
            elif reconstruction_loss_function_name == 'log_cosh':
                self.reconstruction_loss_function=tf.keras.losses.LogCosh(reduction=tf.keras.losses.Reduction.NONE)
            elif reconstruction_loss_function_name == 'huber':
                self.reconstruction_loss_function=tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)