import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Lambda, Reshape, Conv2DTranspose, BatchNormalization, Activation, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

def sampling(args):
    z_mean, z_log_var,latent_dim = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

class SamplingLayer(keras.layers.Layer):
    
    def __init__(self,*args, **kwargs):
        super(SamplingLayer, self).__init__(*args,**kwargs)

    def call(self,args):
        return sampling(args)

class Encoder(keras.layers.Layer):
    def __init__(self,latent_dim, *args, **kwargs):
        super(Encoder, self).__init__(*args,**kwargs)
        self.latent_dim=latent_dim

    def call(self,inputs):
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
        x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
        x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)
        latents=SamplingLayer(name='z')([z_mean, z_log_var,self.latent_dim])
        return [z_mean, z_log_var, latents]


def get_encoder(inputs, latent_dim):
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    return Model(inputs, [z_mean, z_log_var, SamplingLayer(name='z')([z_mean, z_log_var,latent_dim])], name='encoder')

def get_decoder(latent_dim, image_dim,n=0):
    decoder1_inputs = Input(shape=(latent_dim,))

    # Decoder 1
    x = Dense(4*4*512, activation='relu')(decoder1_inputs)
    x = Reshape((4, 4, 512))(x)
    x = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    if image_dim > 32:
        x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
    if image_dim > 64:
        x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
    if image_dim > 128:
        x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
    if image_dim > 256:
        x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
    decoder1_outputs = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    return Model(decoder1_inputs, decoder1_outputs,name='decoder_{}'.format(n))

def get_y_vae_list(latent_dim, input_shape, n_decoders):
    inputs = Input(shape=input_shape, name='encoder_input')
    encoder = get_encoder(inputs, latent_dim)
    #encoder=Encoder(latent_dim,name="encoder")
    image_dim=input_shape[1]
    #outputs1 = decoder1(encoder(inputs)[2])
    decoders=[get_decoder(latent_dim, image_dim,n) for n in range(n_decoders)]
    [z_mean, z_log_var, latents]=encoder(inputs)
    outputs=[d(latents) for d in decoders]+[z_mean, z_log_var]
    y_vae_list = [Model(inputs, [d(latents),z_mean, z_log_var]) for d in decoders]
    return y_vae_list

