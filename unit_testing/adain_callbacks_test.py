import sys
sys.path.append('adain')
sys.path.append('loss')

import matplotlib.pyplot as plt
from adain_callbacks import *
from adain_model import *
from adain_data_helper import *
from loss_net_model import *
from adain_trainer import *

def AdainImageGenerationCallback_test(image_dim=64):
    input_shape=(image_dim, image_dim,3)
    encoder=get_encoder(input_shape)
    loss_net=get_loss_net(input_shape)
    decoder=get_decoder()
    optimizer = keras.optimizers.Adam(learning_rate=1e-5)
    loss_fn = keras.losses.MeanSquaredError()
    style_weight=1.0
    callbacks=[]
    train_dataset= adain_get_dataset_train(batch_size=4,unit_test=True,image_dim=image_dim)
    epochs = 3
    save_path='aaslalkd'
    adain_trainer=AdaInTrainer(encoder, decoder, loss_net, style_weight, optimizer, loss_fn ,callbacks,epochs,train_dataset,save_path)
    test_dataset=adain_get_dataset_test(batch_size=8,image_dim=image_dim)
    image_output_dir= 'exploration/'
    callback=AdainImageGenerationCallback(adain_trainer, test_dataset, image_output_dir)
    callback(0)

def AdainSavingCallback_test(image_dim=64):
    input_shape=(image_dim, image_dim,3)
    encoder=get_encoder(input_shape)
    loss_net=get_loss_net(input_shape)
    decoder=get_decoder()
    optimizer = keras.optimizers.Adam(learning_rate=1e-5)
    loss_fn = keras.losses.MeanSquaredError()
    style_weight=1.0
    callbacks=[]
    train_dataset= adain_get_dataset_train(batch_size=4,unit_test=True,image_dim=image_dim)
    epochs = 3
    save_model_path='../../../../../scratch/jlb638/yvae_models/yvae/adain/'
    adain_trainer=AdaInTrainer(encoder, decoder, loss_net, style_weight, optimizer, loss_fn ,callbacks,epochs,train_dataset,save_model_path)
    test_dataset=adain_get_dataset_test(batch_size=8,image_dim=image_dim)
    image_output_dir= 'exploration/'
    callback=AdainModelSaveCallback(adain_trainer)
    callback(0)
    decoder= tf.saved_model.load(save_model_path+"adain_decoder")

if __name__ == '__main__':
    for dim in [64,128,512]:
        AdainImageGenerationCallback_test(dim)
        AdainSavingCallback_test(dim)