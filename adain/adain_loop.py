import tensorflow as tf
#from tensorflow_examples.models.pix2pix import pix2pix
import sys
sys.path.append('loss')

from adain_trainer import *
from adain_model import *
from adain_data_helper import *
from loss_net_model import *
from adain_callbacks import *
import time
import argparse
from datetime import datetime, timezone
import os
from random import randrange
import json

parser = argparse.ArgumentParser(description='get some args')
parser.add_argument("--epochs",type=int,help="training epochs", default=2)
parser.add_argument("--test",type=bool, default=False)
parser.add_argument("--batch_size", type=int,default=1) 
parser.add_argument("--save_img_parent",type=str,default="/home/jlb638/Desktop/vae_plus/gen_imgs/adain/")
parser.add_argument("--name",type=str,default="adain_{}".format(str(datetime.now(timezone.utc))))
parser.add_argument("--save_model_parent", type=str,default="../../../../../scratch/jlb638/yvae_models/adain/")
#parser.add_argument("--dataset_names",nargs="+",default=["jlbaker361/flickr_humans_10k", "jlbaker361/anime_faces_10k","jlbaker361/ar_rom_bar_ren" ])
parser.add_argument("--content_path",type=str,default="jlbaker361/flickr_humans_10k")
parser.add_argument("--style_path",type=str, default="jlbaker361/anime_faces_10k")
parser.add_argument("--load", type=bool, default=False, help="whether to load previous model if possible")
parser.add_argument("--save", type=bool, default=False, help='whether to save model')
parser.add_argument("--image_dim",type=int, default=128)
parser.add_argument("--interval",type=int,default=10,help='save model every interval # of epochs')
parser.add_argument("--threshold",type=int,default=50,help='epoch threshold for when to start saving')
#parser.add_argument("--latent_dim",type=int, default=32,help='latent dim for encoding')
parser.add_argument("--style_weight",type=float, default=1.0)

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


    print(args)
    OUTPUT_CHANNELS = 3
    start_epoch=0
    input_shape=(args.image_dim,args.image_dim, OUTPUT_CHANNELS)
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        encoder=get_encoder(input_shape)
        loss_net=get_loss_net(input_shape)
        optimizer = keras.optimizers.Adam(learning_rate=1e-5)
        if args.load:
            decoder=tf.saved_model.load(save_model_folder+"adain_decoder")
            with open(save_model_folder+"/meta_data.json","r") as src_file:
                start_epoch=json.load(src_file)["epoch"]

            print("successfully loaded from {} at epoch {}".format(save_model_folder, start_epoch))
        else:
            decoder=get_decoder()
            

    train_dataset= adain_get_dataset_train(batch_size=args.batch_size,image_dim=args.image_dim, content_path=args.content_path, style_path=args.style_path, mirrored_strategy=mirrored_strategy)
    test_dataset= adain_get_dataset_test(batch_size=args.batch_size,image_dim=args.image_dim, content_path=args.content_path, style_path=args.style_path, mirrored_strategy=mirrored_strategy)
    callbacks=[]
    loss_fn = keras.losses.MeanSquaredError()
    trainer=AdaInTrainer(encoder, decoder, loss_net, args.style_weight, optimizer, loss_fn ,callbacks,args.epochs,train_dataset, save_model_folder, start_epoch=start_epoch)
    callbacks=[
        AdainImageGenerationCallback(trainer, test_dataset, save_folder)
    ]
    if args.save:
        callbacks.append(AdainModelSaveCallback(trainer,args.threshold,args.interval,save_model_folder))
    trainer.callbacks=callbacks

    print("begin loop :O")
    trainer.train_loop()

    print("all done :)))")

if __name__=='__main__':
    print("begin")
    print(args)
    objective(None, args)
    print("end")
        