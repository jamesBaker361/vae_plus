import tensorflow as tf
#from tensorflow_examples.models.pix2pix import pix2pix
from yvae_data_helper import *
from yvae_model import *
from yvae_trainer import *
from yvae_callbacks import *
import argparse
from datetime import datetime, timezone
import os
from random import randrange
import json

parser = argparse.ArgumentParser(description='get some args')
parser.add_argument("--epochs",type=int,help="training epochs", default=2)
parser.add_argument("--test",type=bool, default=False)
parser.add_argument("--batch_size", type=int,default=1) 
parser.add_argument("--save_img_parent",type=str,default="/home/jlb638/Desktop/vae_plus/gen_imgs/yvae/")
parser.add_argument("--name",type=str,default="yvae_{}".format(str(datetime.now(timezone.utc))))
parser.add_argument("--save_model_parent", type=str,default="../../../../../scratch/jlb638/yvae_models/yvae/")
parser.add_argument("--dataset_names",nargs="+",default=["jlbaker361/flickr_humans_10k", "jlbaker361/anime_faces_10k" ])
parser.add_argument("--load", type=bool, default=False, help="whether to load previous model if possible")
parser.add_argument("--save", type=bool, default=False, help='whether to save model')
parser.add_argument("--image_dim",type=int, default=128)
parser.add_argument("--interval",type=int,default=10,help='save model every interval # of epochs')
parser.add_argument("--threshold",type=int,default=50,help='epoch threshold for when to start saving')
parser.add_argument("--latent_dim",type=int, default=32,help='latent dim for encoding')
parser.add_argument("--kl_loss_scale",type=float,default=1.0,help='scale of kl_loss for optimizing')
parser.add_argument("--reconstruction_loss_function_name",type=str,default='mse')

args = parser.parse_args()

from tensorflow.python.framework.ops import disable_eager_execution

#disable_eager_execution()

def objective(trial,args):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("tf.test.is_gpu_available() =", tf.test.is_gpu_available())
    save_folder=args.save_img_parent+args.name+"/"
    save_model_folder=args.save_model_parent+args.name+"/"
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(save_model_folder, exist_ok=True)
    n_decoders=len(args.dataset_names)

    print(args)
    OUTPUT_CHANNELS = 3
    start_epoch=0
    input_shape=(args.image_dim,args.image_dim, OUTPUT_CHANNELS)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        optimizer=keras.optimizers.legacy.Adam(learning_rate=0.0001)
        if args.load:
            encoder=keras.models.load_model(save_model_folder+"encoder")
            decoders=[keras.models.load_model(save_model_folder+"decoder_{}".format(d)) for d in range(n_decoders)]
            inputs = encoder.inputs
            [z_mean, z_log_var, latents]=encoder(inputs)
            y_vae_list = [Model(inputs, [d(latents),z_mean, z_log_var]) for d in decoders]

            with open(save_model_folder+"/meta_data.json","r") as src_file:
                start_epoch=json.load(src_file)["epoch"]

            print("successfully loaded from {} at epoch {}".format(save_model_folder, start_epoch),flush=True)

        else:
            y_vae_list=get_y_vae_list(args.latent_dim, input_shape, n_decoders)

    dataset_dict=yvae_get_dataset_train(batch_size=args.batch_size, dataset_names=args.dataset_names, image_dim=args.image_dim,mirrored_strategy=mirrored_strategy)
    test_dataset_dict=yvae_get_dataset_test(batch_size=args.batch_size, dataset_names=args.dataset_names, image_dim=args.image_dim, mirrored_strategy=mirrored_strategy)
    trainer=YVAE_Trainer(y_vae_list, args.epochs,dataset_dict,optimizer,reconstruction_loss_function_name=args.reconstruction_loss_function_name,start_epoch=start_epoch,kl_loss_scale=args.kl_loss_scale)
    callbacks=[
        YvaeImageGenerationCallback(trainer, test_dataset_dict, save_folder, 3)
    ]
    if args.save:
        callbacks.append(YvaeSavingCallback(trainer, save_model_folder, args.threshold, args.interval))
    trainer.callbacks=callbacks
    print("begin loop :O")
    trainer.train_loop()

    print("all done :)))")

if __name__ == '__main__':
    print("begin")
    print(args)
    objective(None, args)
    print("end")