import os
os.environ["XLA_FLAGS"] ="--xla_gpu_cuda_data_dir=/home/jlb638/.conda/envs/fine-tune/lib"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices --tf_xla_cpu_global_jit'
import tensorflow as tf
tf.config.optimizer.set_jit(True)
from yvae_data_helper import *
from yvae_model import *
from yvae_trainer import *
from yvae_callbacks import *
import argparse
from datetime import datetime, timezone
import psutil
import gc
from memory_profiler import profile

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
parser.add_argument("--disable_strategy",help="whether to use mirrored_strategy in trainer",type=bool,default=False)
parser.add_argument("--fine_tuning",type=bool, default=False,help="wheter to use fine tuning training (freezing encoder initially)")
parser.add_argument("--init_lr",type=float,default=0.001,help='lr for adam optimizer')
parser.add_argument("--unfreezing_epoch",type=int,default=-1,help='epoch to unfreeze pretrained encoder for fine tuning')
parser.add_argument("--use_residual",type=bool,default=False)
parser.add_argument("--node",type=str,default='unknown')

args = parser.parse_args()

tf.config.run_functions_eagerly(True)


def objective_unit(trial,args):
    print('eager mode = ', tf.executing_eagerly())
    print('tf.config.experimental.get_synchronous_execution() =',tf.config.experimental.get_synchronous_execution())
    physical_devices= tf.config.list_physical_devices('GPU')
    print("Num physical GPUs Available: ",len(physical_devices))
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    
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

    mirrored_strategy = tf.distribute.MirroredStrategy()
    #mirrored_strategy = tf.distribute.MirroredStrategy()
    start=time.time()
    if args.disable_strategy==False:
        with mirrored_strategy.scope():
            print(mirrored_strategy)
            print("begin mirrored stuff")

            data_start=time.time()
            dataset_dict=yvae_get_dataset_train(batch_size=args.batch_size, dataset_names=args.dataset_names, image_dim=args.image_dim,mirrored_strategy=mirrored_strategy)
            test_dataset_dict=yvae_get_dataset_test(batch_size=args.batch_size, dataset_names=args.dataset_names, image_dim=args.image_dim, mirrored_strategy=mirrored_strategy)
            data_end=time.time()
            print("seeting up data took {} seconds ".format(data_end-data_start))
            
            optimizer_start=time.time()
            optimizer=keras.optimizers.Adam(learning_rate=args.init_lr)
            unfrozen_optimizer=keras.optimizers.Adam(learning_rate=0.00001)
            print("optimizers took {} seconds".format(time.time()-optimizer_start))
            #optimizer=tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
            process = psutil.Process(os.getpid())
            pct=process.memory_percent()
            print('mem pct',pct)
            time.sleep(5)
            gc.collect()
            if args.load:
                print("loading from saved model")
                shared_partial=tf.keras.models.load_model(save_model_folder+SHARED_ENCODER_NAME)
                decoders=[tf.keras.models.load_model(save_model_folder+DECODER_NAME.format(i)) for i in range(n_classes)]
                partials=[tf.keras.models.load_model(save_model_folder+UNSHARED_PARTIAL_ENCODER_NAME.format(i)) for i in range(n_classes)]
                unit_list=load_unit_list(shared_partial, decoders, partials)

                with open(save_model_folder+"/meta_data.json","r") as src_file:
                    start_epoch=json.load(src_file)["epoch"]

                print("successfully loaded from {} at epoch {}".format(save_model_folder, start_epoch))

            else:
                print("not loading from saved")
                encoder_start=time.time()
                print('encoder args',input_shape,args.latent_dim, args.use_residual)
                encoder=get_encoder(input_shape,args.latent_dim, use_residual=args.use_residual)
                encoder_end=time.time()
                print('getting encoder took {} time'.format(encoder_end-encoder_start))
                mid_name=ENCODER_CONV_NAME.format(2)
                unit_list=get_unit_list(input_shape,args.latent_dim,n_classes,encoder,mid_name=mid_name, use_residual=args.use_residual)
                print("unit_list time took {}".format(time.time()-encoder_end))

            mirron_end=time.time()
            print("mirrored stuff took {} seconds".format(mirron_end-start))
    else:
        print(mirrored_strategy)
        print("begin NOT mirrored stuff")
        
        optimizer=keras.optimizers.Adam(learning_rate=args.init_lr)
        unfrozen_optimizer=keras.optimizers.Adam(learning_rate=0.00001)
        print("optimizers took {} seconds".format(time.time()-start))
        #optimizer=tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        if args.load:
            print("loading from saved model")
            shared_partial=tf.keras.models.load_model(save_model_folder+SHARED_ENCODER_NAME)
            decoders=[tf.keras.models.load_model(save_model_folder+DECODER_NAME.format(i)) for i in range(n_classes)]
            partials=[tf.keras.models.load_model(save_model_folder+UNSHARED_PARTIAL_ENCODER_NAME.format(i)) for i in range(n_classes)]
            unit_list=load_unit_list(shared_partial, decoders, partials)

            with open(save_model_folder+"/meta_data.json","r") as src_file:
                start_epoch=json.load(src_file)["epoch"]

            print("successfully loaded from {} at epoch {}".format(save_model_folder, start_epoch))

        else:
            print("not loading from saved")
            encoder_start=time.time()
            encoder=get_encoder(input_shape,args.latent_dim, use_residual=args.use_residual)
            encoder_end=time.time()
            print('getting encoder took {} time'.format(encoder_end-encoder_start))
            mid_name=ENCODER_CONV_NAME.format(2)
            unit_list=get_unit_list(input_shape,args.latent_dim,n_classes,encoder,mid_name=mid_name, use_residual=args.use_residual)
            print("unit_list time took {}".format(time.time()-encoder_end))

        mirrored_strategy=None

        data_start=time.time()
        print("mirrored stuff minus data took {} seconds".format(data_start-start))
        dataset_dict=yvae_get_dataset_train(batch_size=args.batch_size, dataset_names=args.dataset_names, image_dim=args.image_dim,mirrored_strategy=mirrored_strategy)
        test_dataset_dict=yvae_get_dataset_test(batch_size=args.batch_size, dataset_names=args.dataset_names, image_dim=args.image_dim, mirrored_strategy=mirrored_strategy)
        data_end=time.time()
        print("seeting up data took {} seconds ".format(data_end-data_start))
    trainer=VAE_Unit_Trainer(unit_list, args.epochs, dataset_dict=dataset_dict, test_dataset_dict=test_dataset_dict, 
                             optimizer=optimizer, log_dir=log_dir, mirrored_strategy=mirrored_strategy, kl_loss_scale=args.kl_loss_scale,start_epoch=start_epoch,
                             unfreezing_epoch=args.unfreezing_epoch, fine_tuning=args.fine_tuning, unfrozen_optimizer=unfrozen_optimizer )
    callbacks=[
        YvaeImageGenerationCallback(trainer, test_dataset_dict, save_folder, 3)
    ]
    if args.save:
        callbacks.append(YvaeUnitSavingCallback(trainer, save_model_folder, args.threshold, args.interval))
    trainer.callbacks=callbacks
    end=time.time()
    print("seeting up took {} seconds ".format(end-start))
    print("begin loop :O")
    trainer.train_loop()

    print("all done :)))")
    return trial

if __name__ == '__main__':
    print("begin")
    print(args)
    objective_unit(None, args)
    print("end")