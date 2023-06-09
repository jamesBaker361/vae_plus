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
UNSHARED_PARTIAL_ENCODER_NAME='unshared_partial_encoder_{}'
ENCODER_BN_NAME='encoder_bn_{}'
ENCODER_STEM_NAME='encoder_stem_{}'
ENCODER_CONV_NAME='encoder_conv_{}'
DECODER_NAME='decoder_{}'
ENCODER_NAME='encoder'
CLASSIFIER_MODEL='classifier_model'
CLASSIFICATION_HEAD='classification_head'
Z_MEAN='z_mean'
Z_LOG_VAR='z_log_var'
RESNET_CLASSIFIER='resnet_classifier'
MOBILE_NET='mobile'
EFFICIENT_NET='efficient'
VGG='vgg19'

class SoftmaxWithMaxSubtraction(tf.keras.layers.Layer):
    def call(self, inputs):
        max_values = tf.reduce_max(inputs, axis=-1, keepdims=True)
        subtracted_values = tf.subtract(inputs, max_values)
        softmax_output = tf.nn.softmax(subtracted_values, axis=-1)
        return softmax_output

def sampling(args):
    z_mean, z_log_var,latent_dim = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

class SamplingLayer(keras.layers.Layer):
    
    def __init__(self,*args, **kwargs):
        super(SamplingLayer, self).__init__(*args,**kwargs)

    def call(self,args):
        return sampling(args)


def get_encoder(input_shape, latent_dim):
    inputs= Input(shape=input_shape, name=ENCODER_INPUT_NAME)
    x = Conv2D(32, (3, 3), padding='same', name=ENCODER_CONV_NAME.format(0))(inputs)
    x=tf.keras.layers.LeakyReLU()(x)
    x = Conv2D(32, (3, 3), padding='same', name=ENCODER_CONV_NAME.format(1))(x)
    x=tf.keras.layers.LeakyReLU()(x)
    x = BatchNormalization(name=ENCODER_BN_NAME.format(0))(x)
    print('model 51')
    count=2
    bn_count=1
    for dim in [64, 128,128]:
        print(dim)
        x = Conv2D(dim, (3, 3), strides=(2, 2), padding='same',name=ENCODER_CONV_NAME.format(count))(x)
        x=tf.keras.layers.LeakyReLU()(x)
        count+=1
        #x = Conv2D(dim, (3, 3), padding='same', name=ENCODER_CONV_NAME.format(count))(x)
        #x=tf.keras.layers.LeakyReLU()(x)
        #count+=1
        x = BatchNormalization(name=ENCODER_BN_NAME.format(bn_count))(x)
        bn_count+=1
    x = Flatten(name="flatten")(x)
    z_mean = Dense(latent_dim, name=Z_MEAN)(x)
    z_log_var = Dense(latent_dim, name=Z_LOG_VAR)(x)
    return Model(inputs, [z_mean, z_log_var, SamplingLayer(name='z')([z_mean, z_log_var,latent_dim])], name=ENCODER_NAME)

def get_unshared_partial_encoder(input_shape,latent_dim, start_name,mid_name,n):
    '''gets the part of the encoder from layers [start_name:mid_name] (exclusive)
    '''
    encoder=get_encoder(input_shape=input_shape, latent_dim=latent_dim)
    return extract_unshared_partial_encoder(encoder,start_name=start_name, mid_name=mid_name,n=n)

def extract_unshared_partial_encoder(encoder, start_name,mid_name,n):
    layer_names=[layer.name for layer in encoder.layers]
    start_index=layer_names.index(start_name)
    end_index=layer_names.index(mid_name)
    subset=encoder.layers[start_index:end_index]
    return tf.keras.Sequential(subset,name=UNSHARED_PARTIAL_ENCODER_NAME.format(n))

def get_shared_partial(encoder, mid_name, latent_dim):
    '''gets the part of the encoder from layer [mid_name:flatten]
    '''
    layer_names=[layer.name for layer in encoder.layers]
    start_index=layer_names.index(mid_name)
    flatten_index=layer_names.index('flatten')
    input_shape=encoder.layers[start_index].input_shape[1:]
    inputs = Input(shape=input_shape)
    #pretrained_encoder.summary()
    flat_encoder= tf.keras.Sequential(encoder.layers[start_index:flatten_index+1] )
    x=flat_encoder(inputs)
    z_mean=encoder.get_layer(Z_MEAN)(x)
    z_log_var =encoder.get_layer(Z_LOG_VAR)(x)
    return Model(inputs, [z_mean, z_log_var, SamplingLayer(name='z')([z_mean, z_log_var,latent_dim])], name=SHARED_ENCODER_NAME)

    #subset=pretrained_encoder.layers[start_index:]
    #return tf.keras.Sequential(subset)

def get_mixed_pretrained_encoder(input_shape,latent_dim, shared_partial, mid_name,n=0):
    #this is the shared and unshared parts together
    partial=get_unshared_partial_encoder(input_shape,latent_dim, ENCODER_INPUT_NAME,mid_name,n)
    #shared_partial=get_shared_partial(pretrained_encoder, start_name, latent_dim)
    x=partial(partial.input)
    [z_mean, z_log, z]=shared_partial(x)
    return Model(partial.input, [z_mean, z_log, z],name=ENCODER_STEM_NAME.format(n))

def get_unit_list(input_shape,latent_dim,n_classes,encoder, mid_name):
    shared_partial=get_shared_partial(encoder, mid_name, latent_dim)
    encoder_list=[get_mixed_pretrained_encoder(input_shape,latent_dim, shared_partial, mid_name,n) for n in range(n_classes)]
    decoder_list=[get_decoder(latent_dim, input_shape[1],n) for n in range(n_classes)]
    return compile_unit_list(encoder_list,decoder_list)

def build_unit_list_testing(input_shape,latent_dim,n_classes,mid_name):
    encoder=get_encoder(input_shape,latent_dim, latent_dim)
    unit_list=get_unit_list(input_shape,latent_dim,n_classes,encoder, mid_name)
    return unit_list

def get_unit_list_from_creative(pretrained_encoder,n_classes,input_shape,mid_name,latent_dim): #uses pretrained encoder to make unit_list
    shared_partial=get_shared_partial(pretrained_encoder, mid_name, latent_dim)
    pretrained_unshared_partial=extract_unshared_partial_encoder(pretrained_encoder, 
                                                                 start_name=ENCODER_INPUT_NAME,
                                                                 mid_name=mid_name,
                                                                  n=0)
    src_weights=pretrained_unshared_partial.get_weights()

    unshared_partial_list=[]
    for n in range(n_classes):
        unshared_partial=get_unshared_partial_encoder(input_shape, latent_dim, ENCODER_INPUT_NAME,mid_name,n)
        unshared_partial.set_weights(src_weights)
        unshared_partial_list.append(unshared_partial)
    
    encoder_list=[]
    for n,unshared_partial in enumerate(unshared_partial_list):
        inputs= Input(shape=input_shape, name=ENCODER_INPUT_NAME)
        x=unshared_partial(inputs)
        [z_mean, z_log, z]=shared_partial(x)
        encoder = Model(inputs, [z_mean, z_log, z],name=ENCODER_STEM_NAME.format(n))
        encoder_list.append(encoder)

    decoder_list=[get_decoder(latent_dim, input_shape[1],n) for n in range(n_classes)]
    return compile_unit_list(encoder_list,decoder_list)


def load_unit_list(shared_partial, decoder_list, partials):
    encoder_list=[]
    for n in range(len(decoder_list)):
        partial=partials[n]
        partial.summary()
        input_shape=partial.input_shape[1:]
        inputs=Input(shape=input_shape, name=ENCODER_INPUT_NAME)
        x=partial(inputs)
        [z_mean, z_log, z]=shared_partial(x)
        mixed_pretrained_encoder=Model(inputs, [z_mean, z_log, z],name=ENCODER_STEM_NAME.format(n))
        encoder_list.append(mixed_pretrained_encoder)
    return compile_unit_list(encoder_list,decoder_list)



def compile_unit_list(encoder_list,decoder_list):
    unit_list=[]
    for encoder,decoder in zip(encoder_list,decoder_list):
        [z_mean, z_log_var, latents]=encoder(encoder.input)
        unit_list.append(Model(encoder.input, [decoder(latents),z_mean, z_log_var ]))
    return unit_list

def get_decoder(latent_dim, image_dim,n=0):
    decoder1_inputs = Input(shape=(latent_dim,))

    # Decoder 1
    x = Dense(4*4*128)(decoder1_inputs)
    x=tf.keras.layers.LeakyReLU()(x)
    x = Reshape((4, 4, 128))(x)
    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x=tf.keras.layers.LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
    x=tf.keras.layers.LeakyReLU()(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x=tf.keras.layers.LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
    x=tf.keras.layers.LeakyReLU()(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x=tf.keras.layers.LeakyReLU()(x)
    x = BatchNormalization()(x)
    if image_dim > 32:
        x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
        x=tf.keras.layers.LeakyReLU()(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x=tf.keras.layers.LeakyReLU()(x)
        x = BatchNormalization()(x)
    if image_dim > 64:
        x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(x)
        x=tf.keras.layers.LeakyReLU()(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x=tf.keras.layers.LeakyReLU()(x)
        x = BatchNormalization()(x)
    if image_dim > 128:
        x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(x)
        x=tf.keras.layers.LeakyReLU()(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x=tf.keras.layers.LeakyReLU()(x)
        x = BatchNormalization()(x)
    if image_dim > 256:
        x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(x)
        x=tf.keras.layers.LeakyReLU()(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x=tf.keras.layers.LeakyReLU()(x)
        x = BatchNormalization()(x)
    decoder1_outputs = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    return Model(decoder1_inputs, decoder1_outputs,name='decoder_{}'.format(n))

def get_classification_head(latent_dim,n_classes):
    inputs = Input(shape=(latent_dim,))
    x=Dense(latent_dim//4)(inputs)
    x=tf.keras.layers.LeakyReLU()(x)
    x=Dropout(0.2)(x)
    x=Dense(n_classes)(x)
    x=SoftmaxWithMaxSubtraction()(x)
    return Model(inputs, x,name=CLASSIFICATION_HEAD)

def get_classifier_model(latent_dim,input_shape,n_classes):
    encoder = get_encoder(input_shape, latent_dim)
    encoder.build(input_shape)
    classification_head=get_classification_head(latent_dim, n_classes)
    classification_head.build((latent_dim))

    [z_mean, z_log_var, latents]=encoder(encoder.inputs)
    predictions=classification_head(latents)
    return Model(encoder.inputs, predictions, name=CLASSIFIER_MODEL)



def get_y_vae_list(latent_dim, input_shape, n_classes):
    #we use this for creativty and non-unit
    encoder = get_encoder(input_shape, latent_dim)
    inputs=encoder.inputs
    encoder.build(input_shape)
    #encoder=Encoder(latent_dim,name="encoder")
    image_dim=input_shape[1]
    #outputs1 = decoder1(encoder(inputs)[2])
    decoder_list=[get_decoder(latent_dim, image_dim,n) for n in range(n_classes)]
    for dec in decoder_list:
        dec.build((latent_dim))
    [z_mean, z_log_var, latents]=encoder(inputs)
    y_vae_list = [Model(inputs, [d(latents),z_mean, z_log_var]) for d in decoder_list]
    for vae in y_vae_list:
        vae.build(input_shape)
    return y_vae_list

def residual_block(input_tensor, filters, kernel_size):
    # Convolutional layers
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Residual connection
    residual = input_tensor + x
    output = tf.keras.layers.ReLU()(residual)
    return output

def get_resnet_classifier(input_shape, n_classes):
    # Input layer
    inputs = tf.keras.Input(shape=input_shape)

    # First convolutional layer
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Residual blocks
    x = residual_block(x, 64, 3)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 64, 3)

    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Output layer
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs,name=RESNET_CLASSIFIER)
    return model

def get_external_classifier(input_shape,name):
    mapping={
        MOBILE_NET:tf.keras.applications.MobileNetV3Small,
        EFFICIENT_NET:tf.keras.applications.efficientnet.EfficientNetB0,
        VGG: tf.keras.applications.vgg19.VGG19
    }
    model=mapping[name]
    return model(input_shape=input_shape, include_top=False)