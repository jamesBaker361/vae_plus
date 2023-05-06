import sys
sys.path.append('adain')
sys.path.append('loss')

from adain_trainer import *
from adain_model import *
from adain_data_helper import *
from loss_net_model import *

def adain_trainer_test(image_dim=64):
    input_shape=(image_dim, image_dim,3)
    encoder=get_encoder(input_shape)
    loss_net=get_loss_net(input_shape)
    decoder=get_decoder()
    optimizer = keras.optimizers.Adam(learning_rate=1e-5)
    loss_fn = keras.losses.MeanSquaredError()
    style_weight=1.0
    callbacks=[]
    save_path='../../../../scratch/jlb638/yvae_models/adain/'
    train_dataset= adain_get_dataset_train(batch_size=4,unit_test=True,image_dim=image_dim)
    epochs = 3
    trainer=AdaInTrainer(encoder, decoder, loss_net, style_weight, optimizer, loss_fn ,callbacks,epochs,train_dataset, save_path)
    trainer.train_loop()

if __name__=='__main__':
    for dim in [64,128,512]:
        print('========= {} ========='.format(dim))
        adain_trainer_test(dim)
    print("all done :)")