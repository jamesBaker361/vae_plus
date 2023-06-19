import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices --tf_xla_cpu_global_jit'
os.environ["XLA_FLAGS"] ="--xla_gpu_cuda_data_dir=/home/jlb638/.conda/envs/fine-tune/lib"
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

from yvae_parser_setup import *
parser = argparse.ArgumentParser(description='get some args')
add_arguments(parser)
args = parser.parse_args()

#disable_eager_execution()

tf.config.run_functions_eagerly(True)

def objective(trial,args):
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
    n_decoders=len(args.dataset_names)

    print(args)
    OUTPUT_CHANNELS = 3
    start_epoch=0
    input_shape=(args.image_dim,args.image_dim, OUTPUT_CHANNELS)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        GLOBAL_BATCH_SIZE = args.batch_size * mirrored_strategy.num_replicas_in_sync
        optimizer=keras.optimizers.Adam(learning_rate=0.0001)
        #optimizer=tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        if args.load:
            encoder=keras.models.load_model(save_model_folder+ENCODER_NAME)
            decoders=[keras.models.load_model(save_model_folder+DECODER_NAME.format(d)) for d in range(n_decoders)]
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
    trainer=YVAE_Trainer(y_vae_list, args.epochs,dataset_dict,test_dataset_dict,optimizer,reconstruction_loss_function_name=args.reconstruction_loss_function_name,log_dir=log_dir,mirrored_strategy=mirrored_strategy,start_epoch=start_epoch,kl_loss_scale=args.kl_loss_scale,global_batch_size=GLOBAL_BATCH_SIZE)
    callbacks=[
        YvaeImageGenerationCallback(trainer, test_dataset_dict, save_folder, 3)
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