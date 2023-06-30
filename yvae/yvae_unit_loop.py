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
from yvae_parser_setup import *

from random import randrange
import json

parser = argparse.ArgumentParser(description='get some args')
add_arguments(parser)

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

    print("tensorboard command:")
    print("\ttensorboard dev upload --logdir /{}/ --one_shot".format(log_dir))

    OUTPUT_CHANNELS = 3
    start_epoch=0
    input_shape=(args.image_dim,args.image_dim, OUTPUT_CHANNELS)

    mirrored_strategy = tf.distribute.MirroredStrategy(logical_gpus)
    #mirrored_strategy = tf.distribute.MirroredStrategy()
    start=time.time()
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
            if len(args.pretrained_creativity_path)==0:
                encoder=get_encoder(input_shape,args.latent_dim, use_residual=args.use_residual, use_bn=args.use_bn,use_gn=args.use_gn)
            else:
                encoder=tf.keras.models.load_model(args.pretrained_creativity_path)
                print('loaded from creativity!')
            encoder_end=time.time()
            print('getting encoder took {} time'.format(encoder_end-encoder_start))
            mid_name=ENCODER_CONV_NAME.format(2)
            unit_list=get_unit_list(input_shape,args.latent_dim,n_classes,encoder,mid_name=mid_name, use_residual=args.use_residual,use_bn=args.use_bn)
            print("unit_list time took {}".format(time.time()-encoder_end))

        mirron_end=time.time()
        print("mirrored stuff took {} seconds".format(mirron_end-start))
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