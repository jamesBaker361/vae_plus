import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Lambda, Reshape, Conv2DTranspose, BatchNormalization, Activation, UpSampling2D, Layer, Dropout, GlobalAveragePooling2D, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
from retrying import retry
import traceback
import time
import os
import psutil
import numpy as np
import matplotlib.pyplot as plt
import gc
from timeout_decorator import timeout
import retrying

process = psutil.Process(os.getpid())

SHARED_ENCODER_NAME='shared_encoder'
ENCODER_INPUT_NAME='encoder_input'
UNSHARED_PARTIAL_ENCODER_NAME='unshared_partial_encoder_{}'
ENCODER_BN_NAME='encoder_bn_{}'
RESIDUAL_LAYER_NAME='residual_layer_{}'
ENCODER_STEM_NAME='encoder_stem_{}'
ENCODER_CONV_NAME='encoder_conv_{}'
DECODER_NAME='decoder_{}'
ENCODER_NAME='encoder'
CLASSIFIER_MODEL='classifier_model'
CLASSIFICATION_HEAD='classification_head'
Z_MEAN='z_mean'
Z_LOG_VAR='z_log_var'
RESNET_CLASSIFIER='resnet_classifier'

FLATTEN='flatten'

MOBILE_NET='mobile' #ass
MOBILE_LARGE="mobilenet_large"
EFFICIENT_NET='efficient' #ass
EFFICIENT_B7='efficient_b7'
EFFICIENT_B4='efficient_b4'
VGG='vgg19'
XCEPTION='xception'
INCEPTION='inception' #ass
RESNET_50V2='resnet_50v2'
RESNET_152V2='resnet_152v2'
EXTERNAL_MODEL='external_model'
EXTERNAL_NAME_LIST=[ INCEPTION ,MOBILE_NET, 
                    EFFICIENT_NET, VGG, XCEPTION, MOBILE_LARGE, 
                    EFFICIENT_B7, EFFICIENT_B4, RESNET_50V2, RESNET_152V2]


class SoftmaxWithMaxSubtraction(Layer):
    def call(self, inputs):
        max_values = tf.reduce_max(inputs, axis=-1, keepdims=True)
        subtracted_values = tf.subtract(inputs, max_values)
        softmax_output = tf.nn.softmax(subtracted_values, axis=-1)
        return softmax_output

@retrying.retry(stop_max_attempt_number=5)
@timeout(5)
def sampling(args):
    #print('sampling')
    #process = psutil.Process(os.getpid())
    #pct=process.memory_percent()
    #print('mem pct',pct)
    #print(args)
    z_mean, z_log_var,latent_dim = args
    #print('unpacked args')
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    #print('caluclated epsilon')
    return z_mean + K.exp(z_log_var / 2) * epsilon

class SamplingLayer(Layer):
    
    
    def __init__(self,*args, **kwargs):
        super(SamplingLayer, self).__init__(*args,**kwargs)

    def call(self,args):
        ret= sampling(args)
        return ret


def get_encoder(input_shape, latent_dim,use_residual=False):
    def res(x, dim, name):
        if use_residual:
            return ResidualLayer(dim,kernel_size=(3,3), name=name)(x)
        else:
            return x
        
    inputs= Input(shape=input_shape, name=ENCODER_INPUT_NAME)
    x = Conv2D(32, (3, 3), padding='same', name=ENCODER_CONV_NAME.format(0))(inputs)
    x=tf.keras.layers.LeakyReLU()(x)
    x = Conv2D(32, (3, 3), padding='same', name=ENCODER_CONV_NAME.format(1))(x)
    x = BatchNormalization(name=ENCODER_BN_NAME.format(0))(x)
    x=tf.keras.layers.LeakyReLU()(x)
    count=2
    bn_count=1
    for dim in [64, 128,128]:
        x = Conv2D(dim, (3, 3), strides=(2, 2), padding='same',name=ENCODER_CONV_NAME.format(count))(x)
        x=tf.keras.layers.LeakyReLU()(x)
        count+=1
        x = res(x,dim, RESIDUAL_LAYER_NAME.format(count))
        x = Conv2D(dim, (3, 3), padding='same', name=ENCODER_CONV_NAME.format(count))(x)
        x=tf.keras.layers.LeakyReLU()(x)
        x = BatchNormalization(name=ENCODER_BN_NAME.format(bn_count))(x)
        bn_count+=1
        count+=1
        x = res(x,dim, RESIDUAL_LAYER_NAME.format(count))
        #x = ResidualLayer(dim,kernel_size=(3,3), name=RESIDUAL_LAYER_NAME.format(count))(x)
        #x = BatchNormalization(name=ENCODER_BN_NAME.format(bn_count))(x)
        #bn_count+=1
    x = Flatten(name=FLATTEN)(x)
    z_mean = Dense(latent_dim, name=Z_MEAN)(x)
    z_log_var = Dense(latent_dim, name=Z_LOG_VAR)(x)
    print('intermediate layers made')
    z=SamplingLayer(name='z')
    print('sampling layer made :)')
    x=z([z_mean, z_log_var,latent_dim])
    print('functional outputs made')
    return Model(inputs, [z_mean, z_log_var, x], name=ENCODER_NAME)

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
    flatten_index=layer_names.index(FLATTEN)
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

def get_unit_list(input_shape,latent_dim,n_classes,encoder, mid_name, use_residual=False):
    shared_partial=get_shared_partial(encoder, mid_name, latent_dim)
    encoder_list=[get_mixed_pretrained_encoder(input_shape,latent_dim, shared_partial, mid_name,n) for n in range(n_classes)]
    decoder_list=[get_decoder(latent_dim, input_shape[1],n,use_residual=use_residual) for n in range(n_classes)]
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

def get_decoder(latent_dim, image_dim,n=0,use_residual=False):
    def res(x,dim):
        if use_residual:
            return ResidualLayer(dim,kernel_size=(3,3))(x)
        else:
            return x
        
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
    x = res(x, 128)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x=tf.keras.layers.LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
    x=tf.keras.layers.LeakyReLU()(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x=tf.keras.layers.LeakyReLU()(x)
    x = res(x, 64)
    ###x = ResidualLayer(64,kernel_size=(3,3), )(x)
    if image_dim > 32:
        x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
        x=tf.keras.layers.LeakyReLU()(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x=tf.keras.layers.LeakyReLU()(x)
        x = res(x, 64)
        ##x = ResidualLayer(64,kernel_size=(3,3), )(x)
    if image_dim > 64:
        x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(x)
        x=tf.keras.layers.LeakyReLU()(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x=tf.keras.layers.LeakyReLU()(x)
        x = res(x, 32)
        ##x = ResidualLayer(32,kernel_size=(3,3), )(x)
    if image_dim > 128:
        x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(x)
        x=tf.keras.layers.LeakyReLU()(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x=tf.keras.layers.LeakyReLU()(x)
        x = res(x, 32)
        #x = ResidualLayer(32,kernel_size=(3,3), )(x)
    if image_dim > 256:
        x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(x)
        x=tf.keras.layers.LeakyReLU()(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x=tf.keras.layers.LeakyReLU()(x)
        x = res(x, 32)
        #x = ResidualLayer(128,kernel_size=(3,3), )(x)
    decoder1_outputs = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    return Model(decoder1_inputs, decoder1_outputs,name='decoder_{}'.format(n))

def get_classification_head(class_latent_dim,n_classes):
    inputs = Input(shape=(class_latent_dim,))
    x=Dense(n_classes*2)(inputs)
    x=LeakyReLU()(x)
    x=Dropout(0.2)(x)
    x=Dense(n_classes)(x)
    x=SoftmaxWithMaxSubtraction()(x)
    return Model(inputs, x,name=CLASSIFICATION_HEAD)

def get_classifier_model(latent_dim,input_shape,n_classes,class_latent_dim=0):
    encoder = get_encoder(input_shape, latent_dim)
    extracted_encoder=extract_unshared_partial_encoder(encoder=encoder,
                                                       start_name=ENCODER_INPUT_NAME,
                                                       mid_name=FLATTEN,
                                                       n=0)
    model_layers=[
        extracted_encoder
    ]
    if class_latent_dim<1:
        class_latent_dim=extracted_encoder.output_shape[-1]
    else:
        model_layers+=[
            Dense(class_latent_dim),
            LeakyReLU(),
            Dropout(0.2)
            ]
    model_layers+=[
        GlobalAveragePooling2D(),
        get_classification_head(class_latent_dim, n_classes)
    ]
    return tf.keras.Sequential(model_layers,name=CLASSIFIER_MODEL)



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

class ResidualLayer(keras.layers.Layer):
    def __init__(self, filters, kernel_size, *args, **kwargs):
        super(ResidualLayer, self).__init__(*args, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv1 = tf.keras.layers.Conv2D(self.filters, self.kernel_size, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.leaky_relu1=tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(self.filters, self.kernel_size, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.leaky_relu2=tf.keras.layers.ReLU()

    def build(self, input_shape):
        if len(input_shape)==4:
            input_shape=input_shape[1:]
        self.conv1 = tf.keras.layers.Conv2D(self.filters, self.kernel_size, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.leaky_relu1=tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(self.filters, self.kernel_size, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.leaky_relu2=tf.keras.layers.ReLU()

        if input_shape[-1] != self.filters:
            self.shortcut = tf.keras.layers.Conv2D(self.filters, kernel_size=(1, 1), padding='same')
        else:
            self.shortcut = tf.identity

        #print('LOOK AT ME HELLO input_shape is ',input_shape)
        inputs=Input(shape=input_shape,name='res_layer inputs')
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.leaky_relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)

        shortcut = self.shortcut(inputs)
        x = tf.keras.layers.add([x, shortcut])
        x=self.leaky_relu2(x)

        self.internal_model=keras.Model(inputs, x)

    def call(self, inputs, *args, **kwargs):
        return self.internal_model(inputs,*args, **kwargs)





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
    x = GlobalAveragePooling2D()(x)

    # Output layer
    outputs = Dense(n_classes, activation='softmax')(x)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs,name=RESNET_CLASSIFIER)
    return model

class PreprocessingLayer(Layer):
    def __init__(self, preprocessing_function):
        super().__init__()
        self.preprocessing_function=preprocessing_function

    def call(self,inputs):
        return self.preprocessing_function(inputs)


def get_external_classifier(input_shape,external_name,n_classes,class_latent_dim=0):
    mapping={
        RESNET_50V2: tf.keras.applications.resnet_v2.ResNet50V2,
        RESNET_152V2: tf.keras.applications.resnet_v2.ResNet152V2,
        MOBILE_LARGE: tf.keras.applications.MobileNetV3Large,
        MOBILE_NET:tf.keras.applications.MobileNetV3Small,
        EFFICIENT_NET:tf.keras.applications.efficientnet.EfficientNetB0,
        EFFICIENT_B4:tf.keras.applications.efficientnet.EfficientNetB4,
        EFFICIENT_B7:tf.keras.applications.efficientnet.EfficientNetB7,
        VGG: tf.keras.applications.vgg19.VGG19,
        XCEPTION: tf.keras.applications.xception.Xception,
        INCEPTION:tf.keras.applications.InceptionResNetV2
    }

    preprocessing_mapping={
        RESNET_50V2: tf.keras.applications.resnet_v2.preprocess_input,
        RESNET_152V2: tf.keras.applications.resnet_v2.preprocess_input,
        MOBILE_LARGE: tf.keras.applications.mobilenet_v3.preprocess_input,
        MOBILE_NET:tf.keras.applications.mobilenet_v3.preprocess_input,
        EFFICIENT_NET:tf.keras.applications.efficientnet.preprocess_input,
        EFFICIENT_B4:tf.keras.applications.efficientnet.preprocess_input,
        EFFICIENT_B7:tf.keras.applications.efficientnet.preprocess_input,
        VGG: tf.keras.applications.vgg19.preprocess_input,
        XCEPTION: tf.keras.applications.xception.preprocess_input,
        INCEPTION:tf.keras.applications.inception_resnet_v2.preprocess_input
    }



    pretrained_model=mapping[external_name](input_shape=input_shape, include_top=False)
    preprocessing_layer=PreprocessingLayer(preprocessing_mapping[external_name])
    pretrained_model.summary()
    external= tf.keras.Sequential([
        preprocessing_layer,
        pretrained_model
        ],name=EXTERNAL_MODEL)
    external.build((None, *input_shape))
    external.trainable=False
    model_layers=[
        external
    ]
    if class_latent_dim<1:
        class_latent_dim=external.output_shape[-1]
    else:
        model_layers+=[
            Dense(class_latent_dim),
            LeakyReLU(),
            Dropout(0.2)
            ]
    model_layers+=[
        GlobalAveragePooling2D(),
        get_classification_head(class_latent_dim, n_classes)
    ]
    return tf.keras.Sequential(model_layers,name=CLASSIFIER_MODEL)