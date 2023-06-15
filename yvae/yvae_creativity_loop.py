import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import tensorflow as tf
tf.config.optimizer.set_jit(True)
from yvae_data_helper import *
from yvae_model import *
from yvae_trainer import *
from yvae_callbacks import *
import argparse
from datetime import datetime, timezone
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
parser.add_argument("--log_dir_parent",type=str,default="logs/")
parser.add_argument("--use_strategy",help="whether to use mirrored_strategy in trainer",type=bool,default=False)
parser.add_argument("--creativity_lambda",type=float,default=0.5, help='coefficient on creativity loss' )
parser.add_argument("--pretrained_classifier_path",type=str,help="path to load pretrained_classifier")
parser.add_argument("--use_bn", type=bool, default=False, help='whether to use bn in encoder/decoder')
parser.add_argument("--node",type=str,default='unknown',help='which node this is running on')

args = parser.parse_args()

#disable_eager_execution()

tf.config.run_functions_eagerly(True)

def objective(trial,args):
    physical_devices= tf.config.list_physical_devices('GPU')
    print("Num physical GPUs Available: ",len(physical_devices))
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except:
            print("could not inititate device")
    
    logical_gpus = tf.config.list_logical_devices('GPU')
    print("Logical GPUs ", len(logical_gpus))
    print("tf.test.is_gpu_available() =", tf.test.is_gpu_available())
    save_folder=args.save_img_parent+args.name+"/"
    save_model_folder=args.save_model_parent+args.name+"/"
    log_dir=args.log_dir_parent+args.name
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(save_model_folder, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    n_classes=len(args.dataset_names)

    print(args)
    OUTPUT_CHANNELS = 3
    start_epoch=0
    input_shape=(args.image_dim,args.image_dim, OUTPUT_CHANNELS)

    mirrored_strategy = tf.distribute.MirroredStrategy(logical_gpus, cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    with mirrored_strategy.scope():
        GLOBAL_BATCH_SIZE = args.batch_size * mirrored_strategy.num_replicas_in_sync
        optimizer=keras.optimizers.Adam(learning_rate=0.0001)
        pretrained_classifier=keras.models.load_model(args.pretrained_classifier_path)

        #optimizer=tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        if args.load:
            encoder=keras.models.load_model(save_model_folder+ENCODER_NAME)
            decoder=keras.models.load_model(save_model_folder+DECODER_NAME.format(0))
            inputs = encoder.inputs
            [z_mean, z_log_var, latents]=encoder(inputs)
            y_vae_list = [Model(inputs, [decoder(latents),z_mean, z_log_var])]

            with open(save_model_folder+"/meta_data.json","r") as src_file:
                start_epoch=json.load(src_file)["epoch"]

            print("successfully loaded from {} at epoch {}".format(save_model_folder, start_epoch),flush=True)

        else:
            y_vae_list=get_y_vae_list(args.latent_dim, input_shape, 1)

    dataset_list=yvae_creativity_get_dataset_train(batch_size=args.batch_size,
                                                   dataset_names=args.dataset_names,
                                                   image_dim=args.image_dim,
                                                   mirrored_strategy=mirrored_strategy)
    test_dataset_dict={}
    trainer=VAE_Creativity_Trainer(y_vae_list, args.epochs,{}, test_dataset_dict, optimizer,
                                   dataset_list,log_dir,mirrored_strategy, args.kl_loss_scale,
                                   callbacks=[],
                                   start_epoch=start_epoch,
                                   global_batch_size=GLOBAL_BATCH_SIZE,
                                   pretrained_classifier=pretrained_classifier,
                                   creativity_lambda=args.creativity_lambda,
                                   n_classes=n_classes
                                   )
    callbacks=[
        #YvaeImageGenerationCallback(trainer, test_dataset_dict, save_folder, 3,enable_style_transfer=False)
    ]
    if args.save:
        callbacks.append(YvaeSavingCallback(trainer, save_model_folder, args.threshold, args.interval))
    trainer.callbacks=callbacks
    print("begin loop :O")
    trainer.train_loop()

    print("all done :)))")
    return trial

if __name__ == '__main__':
    print("begin")
    print(args)
    objective(None, args)
    print("end")