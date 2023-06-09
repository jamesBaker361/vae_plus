import os
import sys
sys.path.append(os.getcwd())
sys.path.append('yvae')

import matplotlib.pyplot as plt
from yvae_model import *

def classification_head_test(latent_dim=32, n_classes=4):
    classification_head=get_classification_head(latent_dim, n_classes)
    preds=classification_head(tf.random.normal((4,latent_dim)))
    print(tf.shape(preds))

def classification_model_test(latent_dim=8, input_shape=(32,32,3), n_classes=3):
    classifier=get_classifier_model(latent_dim,input_shape,n_classes)
    img=tf.random.normal((2,*input_shape))
    preds=classifier(img)
    print(preds)

def get_unshared_partial_encoder_test(input_shape=(32,32,3), latent_dim=8, start_layer='encoder_input',end_layer='encoder_conv_2'):
    unshared_partial_encoder=get_unshared_partial_encoder(input_shape, latent_dim,start_layer, end_layer,0)
    unshared_partial_encoder.summary()

def get_shared_partial_test(input_shape=(32,32,3), latent_dim=8, start_layer='encoder_conv_2'):
    encoder=get_encoder(input_shape, latent_dim)
    partial=get_shared_partial(encoder, start_layer, latent_dim)
    print([layer.name for layer in partial.layers])
    partial.summary()

def get_mixed_pretrained_encoder_test(input_shape=(32,32,3), latent_dim=8, start_name='encoder_conv_2'):
    encoder=get_encoder(input_shape, latent_dim)
    shared_partial=get_shared_partial(encoder, start_name, latent_dim)
    mixed=get_mixed_pretrained_encoder(input_shape, latent_dim, shared_partial, start_name)
    mixed.build((None, *input_shape))
    mixed.summary()

def get_unit_list_test(input_shape=(32,32,3), latent_dim=8, start_name='encoder_conv_2',n_classes=3):
    pretrained_encoder=get_encoder(input_shape, latent_dim=latent_dim)
    unit_list=get_unit_list(input_shape,latent_dim,n_classes,pretrained_encoder, start_name)
    unit_list[0].summary()
    unit_list[0](tf.random.normal((1,*input_shape)))

def get_unit_list_from_creative_test(input_shape=(32,32,3), latent_dim=8, start_name='encoder_conv_2',n_classes=3):
    pretrained_encoder=get_encoder(input_shape, latent_dim)
    unit_list=get_unit_list_from_creative(pretrained_encoder,n_classes,input_shape,start_name,latent_dim)
    unit_list[0].summary()
    unit_list[0](tf.random.normal((1,*input_shape)))

def get_resnet_classifier_test(input_shape=(32,32,3), n_classes=3):
    resnet_classifier=get_resnet_classifier(input_shape, n_classes)
    resnet_classifier.summary()
    resnet_classifier(tf.random.normal((1,*input_shape)))

def get_external_classifier_test(input_shape=(64,64,3)):
    for name in [MOBILE_NET, EFFICIENT_NET, VGG]:
        external_classifier=get_external_classifier(input_shape=input_shape,name=name)
        external_classifier(tf.random.normal((1,*input_shape)))




if __name__ =='__main__':
    classification_head_test()
    classification_model_test()
    get_shared_partial_test()
    get_unshared_partial_encoder_test()
    get_mixed_pretrained_encoder_test()
    get_unit_list_test()
    get_unit_list_from_creative_test()
    get_resnet_classifier_test()
    get_external_classifier_test()
    print("all done :)))")
