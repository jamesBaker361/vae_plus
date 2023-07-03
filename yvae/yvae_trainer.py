from yvae_model import *
import numpy as np
import tensorflow as tf
import time
import sys
sys.path.append('evaluation')
from fid_src import *

TRAIN='/train'
TEST='/test'
TEST_INTERVAL=10
FID_BATCH_SIZE=4
TRAIN_LOSS='train_loss'
TEST_LOSS='test_loss'
TRAIN_RECONSTRUCTION_LOSS='train_reconstruction_loss'
TEST_RECONSTRUCTION_LOSS='test_reconstruction_loss'
TRAIN_CREATIVITY_LOSS='train_creativity_loss'
TEST_CREATIVITY_LOSS='test_creativity_loss'
TRAIN_GEN_FID='train_gen_fid_{}'
TRAIN_TRANSFER_FID='train_transfer_fid_{}_to_{}'
TEST_GEN_FID='test_gen_fid_{}'
TEST_TRANSFER_FID='test_transfer_fid_{}_to_{}'

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
    def __init__(self,vae_list,epochs,dataset_dict,test_dataset_dict,optimizer,global_batch_size,log_dir,mirrored_strategy,kl_loss_scale,data_augmentation,callbacks=[],start_epoch=0):
        self.vae_list=vae_list
        self.decoders=[vae_list[i].get_layer(DECODER_NAME.format(i)) for i in range(len(vae_list))]
        self.epochs=epochs
        self.dataset_names=[k for k in dataset_dict.keys()]
        self.dataset_list=[v for v in dataset_dict.values()]
        self.test_dataset_list=[v for v in test_dataset_dict.values()]
        self.dataset_dict=dataset_dict
        self.test_dataset_dict=test_dataset_dict
        self.optimizer=optimizer
        self.callbacks=callbacks
        self.start_epoch=start_epoch
        self.kl_loss_scale=kl_loss_scale
        self.log_dir=log_dir
        self.data_augmentation=data_augmentation
        
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
                self.total_loss=None
                self.data_augmenter= tf.keras.Sequential(
                    [tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"), tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),]
                    )
                
        else:
            self.reconstruction_loss_function=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
            self.train_loss = tf.keras.metrics.Mean(TRAIN_LOSS, dtype=tf.float32)
            self.train_reconstruction_loss= tf.keras.metrics.Mean(TRAIN_RECONSTRUCTION_LOSS, dtype=tf.float32)
            self.test_loss = tf.keras.metrics.Mean(TEST_LOSS, dtype=tf.float32)
            self.test_reconstruction_loss= tf.keras.metrics.Mean(TEST_RECONSTRUCTION_LOSS, dtype=tf.float32)
            self.summary_writer = tf.summary.create_file_writer(log_dir)
            self.compute_kl_loss=get_compute_kl_loss(kl_loss_scale, global_batch_size)
            self.total_loss=None
            self.data_augmenter= tf.keras.Sequential(
                    [tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"), tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),]
                    )
            
        self.train_metrics={
            TRAIN_LOSS:self.train_loss,
            TRAIN_RECONSTRUCTION_LOSS:self.train_reconstruction_loss
        }
        self.test_metrics={
            TEST_LOSS:self.test_loss,
            TEST_RECONSTRUCTION_LOSS: self.test_reconstruction_loss
        }

    #@tf.function
    def train_step(self,batch,vae):
        if self.data_augmentation:
            batch=self.data_augmenter(batch)
        with tf.GradientTape() as tape:
            [reconstruction,z_mean, z_log_var]=vae(batch)
            reconstruction_loss =self.reconstruction_loss_function(batch, reconstruction)
            _total_loss=self.compute_kl_loss(z_mean,z_log_var) +reconstruction_loss
        grads = tape.gradient(_total_loss, vae.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, vae.trainable_weights))
        self.train_loss(_total_loss)
        self.train_reconstruction_loss(reconstruction_loss)
        return _total_loss
    
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
    
    #@tf.function
    def test_step(self,batch,vae):
        [reconstruction,z_mean, z_log_var]=vae(batch)
        reconstruction_loss =self.reconstruction_loss_function(batch, reconstruction)
        total_loss=self.compute_kl_loss(z_mean,z_log_var)
        self.test_loss(total_loss)
        self.test_reconstruction_loss(reconstruction_loss)
        return total_loss

    def epoch_setup(self,e):
        pass

    def epoch_end(self,e):
        pass

    def test_epoch(self,e):
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
                print("\t test {} : {}".format(name, metric.result()))

    def train_loop(self):
        print('train loop begin')
        for e in range(self.start_epoch,self.epochs):
            self.epoch_setup(e)
            start = time.time()
            for d,dataset in enumerate(self.dataset_list):
                vae=self.vae_list[d]
                for batch in dataset:
                    if self.mirrored_strategy is None:
                        total_loss=self.train_step(batch,vae)
                    else:
                        total_loss=self.distributed_train_step(batch,vae)
                    
            #print([ep.numpy() for ep in epoch_losses])
            #print('epoch {} loss: {}'.format(e,self.train_loss.result()))
            print ('\nTime taken for epoch {} is {} sec\n'.format(e,time.time()-start))
            with self.summary_writer.as_default():
                for name,metric in self.train_metrics.items():
                    tf.summary.scalar(name, metric.result(), step=e)
                    print("\t train {} : {}".format(name, metric.result()))
            for callback in self.callbacks:
                callback(e)
            if e%TEST_INTERVAL==0:
                self.test_epoch(e)
            self.epoch_end(e)
        self.test_epoch(e)
        for callback in self.callbacks:
            callback(e)
    
    
    def generate_images(self,batch_size):
        noise_shape=self.decoders[0].input_shape[1:]
        noise=tf.random.normal((batch_size, *noise_shape))
        return [decoder(noise) for decoder in self.decoders]

    def generate_images_calculate_fid(self,batch_size):
        generated=self.generate_images(batch_size)


class VAE_Unit_Trainer(VAE_Trainer):
    def __init__(self,vae_list,epochs,dataset_dict,test_dataset_dict,optimizer,
                 log_dir='',mirrored_strategy=None,kl_loss_scale=1.0,callbacks=[],
                 start_epoch=0,global_batch_size=4, fine_tuning=False,unfreezing_epoch=0, 
                 unfrozen_optimizer=None, data_augmentation=False,
                 fid_batch_size=4, fid_interval=-1):
        super().__init__(vae_list,epochs,dataset_dict,test_dataset_dict,optimizer,log_dir=log_dir,mirrored_strategy=mirrored_strategy ,kl_loss_scale=kl_loss_scale,callbacks=callbacks,start_epoch=start_epoch,global_batch_size=global_batch_size,data_augmentation=data_augmentation)
        vae_list[0].summary()
        self.shared_partial=vae_list[0].get_layer(ENCODER_STEM_NAME.format(0)).get_layer(SHARED_ENCODER_NAME)
        self.partials=[vae_list[i].get_layer(ENCODER_STEM_NAME.format(i)).get_layer(UNSHARED_PARTIAL_ENCODER_NAME.format(i)) for i in range(len(vae_list))]
        self.unfreezing_epoch=unfreezing_epoch
        self.fine_tuning = fine_tuning
        self.unfrozen_optimizer=unfrozen_optimizer
        self.fid_batch_size=fid_batch_size
        self.fid_interval=fid_interval
        if self.fine_tuning:
            self.shared_partial.trainable=False
            for p in self.partials:
                p.trainable=False
        if mirrored_strategy is not None:
            with mirrored_strategy.scope():
                self.setup_fid_metrics()
        else:
            self.setup_fid_metrics()
        
        self.fid_metrics={
            **self.test_gen_fid_dict,
            **self.test_transfer_fid_dict
        }
        self.statistics={}
        for name,dataset in test_dataset_dict.items():
            images=tf.concat([d for d in dataset],axis=0)[:self.fid_batch_size]
            input_shape=(128,128,3)
            mu,sig=calculate_mu_sig(input_shape,images)
            self.statistics[name]=(mu,sig)

    def train_loop(self):
        super().train_loop()
        if self.fid_interval!=-1:
            self.calculate_fid(self.epochs)
    
            

    def setup_fid_metrics(self):
        self.test_gen_fid_dict={
                TEST_GEN_FID.format(name) : tf.keras.metrics.Mean(TEST_GEN_FID.format(name), dtype=tf.float32) for name in self.dataset_names
        }
        self.test_transfer_fid_dict={}
        for x in range(len(self.dataset_names)):
            for y in range(len(self.dataset_names)):
                if y==x:
                    continue
                src=self.dataset_names[x] #initial domain
                target=self.dataset_names[y] #target domain (style)
                self.test_transfer_fid_dict[TEST_TRANSFER_FID.format(src,target)]= tf.keras.metrics.Mean(TEST_TRANSFER_FID.format(src,target), dtype=tf.float32)

    def epoch_setup(self,e):
        if self.fine_tuning and self.unfreezing_epoch<=e:
            self.optimizer=self.unfrozen_optimizer
            self.shared_partial.trainable=True
            for p in self.partials:
                p.trainable=True

    def epoch_end(self,e):
        if e%self.fid_interval==0 and self.fid_interval!=-1:
            self.calculate_fid(e)

    def calculate_fid(self,e):
        generated_images_list=self.generate_images(self.fid_batch_size)
        input_shape=(128,128,3) #the inceptionNet cant have images with dim that are too small, so they have to be expanded
        for x in range(len(self.dataset_names)):
            src_name=self.dataset_names[x]
            mu1,sig1=self.statistics[src_name]
            generated_images=generated_images_list[x]
            mu2,sig2=calculate_mu_sig(input_shape,generated_images)
            test_gen_fid=self.test_gen_fid_dict[TEST_GEN_FID.format(src_name)]
            fid_score=calculate_fid(input_shape, mu1,sig1,mu2,sig2)
            test_gen_fid(fid_score)
            dataset=self.test_dataset_dict[src_name]
            images=tf.concat([d for d in dataset],axis=0)[:self.fid_batch_size]
            style_transferred_images_list=self.style_transfer(images,x)
            for y in range(len(self.dataset_names)):
                if y==x:
                    continue
                target_name=self.dataset_names[y]
                test_transfer_fid=self.test_transfer_fid_dict[TEST_TRANSFER_FID.format(src_name,target_name)]
                style_transferred_images=style_transferred_images_list[y]
                mu1,sig1=self.statistics[target_name]
                mu2,sig2=calculate_mu_sig(input_shape, style_transferred_images)
                fid_score=calculate_fid(input_shape, mu1,sig1,mu2,sig2)
                test_transfer_fid(fid_score)
        with self.summary_writer.as_default():
            for name,metric in self.fid_metrics.items():
                tf.summary.scalar(name, metric.result(), step=e)
                print("\t fid {} : {}".format(name, metric.result()))
        
                

    def style_transfer(self,images,n):
        encoder=self.vae_list[n].get_layer(ENCODER_STEM_NAME.format(n))
        [latents,_,__]=encoder(images)
        ret=[]
        for i,decoder in enumerate(self.decoders):
            ret.append(decoder(latents))
        return ret

class YVAE_Trainer(VAE_Trainer):
    def __init__(self,y_vae_list,epochs,dataset_dict, test_dataset_dict,optimizer,reconstruction_loss_function_name='mse',log_dir='', mirrored_strategy=None,kl_loss_scale=1.0,callbacks=[],start_epoch=0,global_batch_size=4, data_augmentation=False):
        super().__init__(y_vae_list,epochs,dataset_dict,test_dataset_dict,optimizer,log_dir=log_dir,mirrored_strategy=mirrored_strategy ,kl_loss_scale=kl_loss_scale,callbacks=callbacks,start_epoch=start_epoch,global_batch_size=global_batch_size,data_augmentation=data_augmentation)
        self.encoder=y_vae_list[0].get_layer(ENCODER_NAME)
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

class VAE_Creativity_Trainer(YVAE_Trainer):
    def __init__(self,vae_list,epochs,dataset_dict,test_dataset_dict,optimizer,dataset_list,log_dir='',mirrored_strategy=None,kl_loss_scale=1.0,callbacks=[],start_epoch=0, global_batch_size=4,pretrained_classifier=None, creativity_lambda=1.0,n_classes=2,data_augmentation=False):
        super().__init__(vae_list,epochs,dataset_dict=dataset_dict,test_dataset_dict=test_dataset_dict,optimizer=optimizer,log_dir=log_dir,mirrored_strategy=mirrored_strategy ,kl_loss_scale=kl_loss_scale,callbacks=callbacks,start_epoch=start_epoch,global_batch_size=global_batch_size, data_augmentation=data_augmentation)
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