import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Lambda, Reshape, Conv2DTranspose, BatchNormalization, Activation, UpSampling2D, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

SHARED_ENCODER_NAME='shared_encoder'
ENCODER_INPUT_NAME='encoder_input'
PARTIAL_ENCODER_NAME='partial_encoder_{}'
ENCODER_BN_NAME='encoder_bn_{}'
ENCODER_STEM_NAME='encoder_stem_{}'
ENCODER_CONV_NAME='encoder_conv_{}'
DECODER_NAME='decoder_{}'
ENCODER_NAME='encoder'
CLASSIFIER_MODEL='classifier_model'
CLASSIFICATION_HEAD='classification_head'
Z_MEAN='z_mean'
Z_LOG_VAR='z_log_var'

def sampling(args):
    z_mean, z_log_var,latent_dim = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

class SamplingLayer(keras.layers.Layer):
    
    def __init__(self,*args, **kwargs):
        super(SamplingLayer, self).__init__(*args,**kwargs)

    def call(self,args):
        return sampling(args)


def get_encoder(inputs, latent_dim):
    x = Conv2D(32, (3, 3), padding='same', activation='relu',name=ENCODER_CONV_NAME.format(0))(inputs)
    x = Conv2D(32, (3, 3), padding='same', activation='relu',name=ENCODER_CONV_NAME.format(1))(x)
    x = BatchNormalization(name=ENCODER_BN_NAME.format(0))(x)
    count=2
    bn_count=1
    for dim in [64, 128,256, 512]:
        x = Conv2D(dim, (3, 3), strides=(2, 2), padding='same', activation='relu',name=ENCODER_CONV_NAME.format(count))(x)
        count+=1
        x = Conv2D(dim, (3, 3), padding='same', activation='relu',name=ENCODER_CONV_NAME.format(count))(x)
        count+=1
        x = BatchNormalization(name=ENCODER_BN_NAME.format(bn_count))(x)
        bn_count+=1
    x = Flatten(name="flatten")(x)
    z_mean = Dense(latent_dim, name=Z_MEAN)(x)
    z_log_var = Dense(latent_dim, name=Z_LOG_VAR)(x)
    return Model(inputs, [z_mean, z_log_var, SamplingLayer(name='z')([z_mean, z_log_var,latent_dim])], name=ENCODER_NAME)

def get_partial_encoder(input_shape, latent_dim, start_name,end_name,n):
    '''gets the part of the encoder from layers [start_name:end_name] (exclusive)
    '''
    inputs = Input(shape=input_shape, name=ENCODER_INPUT_NAME)
    encoder=get_encoder(inputs, latent_dim)
    layer_names=[layer.name for layer in encoder.layers]
    start_index=layer_names.index(start_name)
    end_index=layer_names.index(end_name)
    subset=encoder.layers[start_index:end_index]
    return tf.keras.Sequential(subset,name=PARTIAL_ENCODER_NAME.format(n))

def get_shared_partial(pretrained_encoder, start_name, latent_dim):
    '''gets the part of the encoder from layer [start_name:]
    '''
    layer_names=[layer.name for layer in pretrained_encoder.layers]
    start_index=layer_names.index(start_name)
    flatten_index=layer_names.index('flatten')
    input_shape=pretrained_encoder.layers[start_index].input_shape[1:]
    inputs = Input(shape=input_shape)
    #pretrained_encoder.summary()
    flat_encoder= tf.keras.Sequential( pretrained_encoder.layers[start_index:flatten_index+1] )
    x=flat_encoder(inputs)
    z_mean=pretrained_encoder.get_layer(Z_MEAN)(x)
    z_log_var =pretrained_encoder.get_layer(Z_LOG_VAR)(x)
    return Model(inputs, [z_mean, z_log_var, SamplingLayer(name='z')([z_mean, z_log_var,latent_dim])], name=SHARED_ENCODER_NAME)

    #subset=pretrained_encoder.layers[start_index:]
    #return tf.keras.Sequential(subset)

def get_mixed_pretrained_encoder(input_shape, latent_dim, shared_partial, start_name,n=0):
    #this is the shared and unshared parts together
    partial=get_partial_encoder(input_shape, latent_dim, ENCODER_INPUT_NAME,start_name,n)
    #shared_partial=get_shared_partial(pretrained_encoder, start_name, latent_dim)
    x=partial(partial.input)
    [z_mean, z_log, z]=shared_partial(x)
    return Model(partial.input, [z_mean, z_log, z],name=ENCODER_STEM_NAME.format(n))

def get_unit_list(input_shape,latent_dim,n_classes,shared_partial, start_name):
    encoders=[get_mixed_pretrained_encoder(input_shape, latent_dim, shared_partial, start_name,n) for n in range(n_classes)]
    decoders=[get_decoder(latent_dim, input_shape[1],n) for n in range(n_classes)]
    return compile_unit_list(encoders,decoders)

def load_unit_list(shared_partial, decoders, partials):
    encoders=[]
    for n in range(len(decoders)):
        partial=partials[n]
        partial.summary()
        input_shape=partial.input_shape
        inputs=Input(shape=input_shape, name=ENCODER_INPUT_NAME)
        x=partial(inputs)
        [z_mean, z_log, z]=shared_partial(x)
        mixed_pretrained_encoder=Model(inputs, [z_mean, z_log, z],name=ENCODER_STEM_NAME.format(n))
        encoders.append(mixed_pretrained_encoder)
    return compile_unit_list(encoders,decoders)



def compile_unit_list(encoders,decoders):
    unit_list=[]
    for encoder,decoder in zip(encoders,decoders):
        [z_mean, z_log_var, latents]=encoder(encoder.input)
        unit_list.append(Model(encoder.input, [decoder(latents),z_mean, z_log_var ]))
    return unit_list

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
    decoder1_outputs = Conv2D(3, (3, 3), activation='tanh', padding='same')(x)
    return Model(decoder1_inputs, decoder1_outputs,name='decoder_{}'.format(n))

def get_classification_head(latent_dim,n_classes):
    inputs = Input(shape=(latent_dim,))
    x=Dense(latent_dim//4,activation='relu')(inputs)
    x=Dropout(0.1)(x)
    x=Dense(n_classes, activation='softmax')(x)
    return Model(inputs, x,name=CLASSIFICATION_HEAD)

def get_classifier_model(latent_dim,input_shape,n_classes):
    inputs = Input(shape=input_shape, name=ENCODER_INPUT_NAME)
    encoder = get_encoder(inputs, latent_dim)
    encoder.build(input_shape)
    classification_head=get_classification_head(latent_dim, n_classes)
    classification_head.build((latent_dim))

    [z_mean, z_log_var, latents]=encoder(inputs)
    predictions=classification_head(latents)
    return Model(inputs, predictions, name=CLASSIFIER_MODEL)



def get_y_vae_list(latent_dim, input_shape, n_decoders):
    inputs = Input(shape=input_shape, name=ENCODER_INPUT_NAME)
    encoder = get_encoder(inputs, latent_dim)
    encoder.build(input_shape)
    #encoder=Encoder(latent_dim,name="encoder")
    image_dim=input_shape[1]
    #outputs1 = decoder1(encoder(inputs)[2])
    decoders=[get_decoder(latent_dim, image_dim,n) for n in range(n_decoders)]
    for dec in decoders:
        dec.build((latent_dim))
    [z_mean, z_log_var, latents]=encoder(inputs)
    y_vae_list = [Model(inputs, [d(latents),z_mean, z_log_var]) for d in decoders]
    for vae in y_vae_list:
        vae.build(input_shape)
    return y_vae_list

