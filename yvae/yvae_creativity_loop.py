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

    print("tensorboard command:")
    print("\ttensorboard dev upload --logdir {}/ --one_shot".format(log_dir))

    mirrored_strategy = tf.distribute.MirroredStrategy(logical_gpus)
    start=time.time()
    with mirrored_strategy.scope():
        GLOBAL_BATCH_SIZE = args.batch_size * mirrored_strategy.num_replicas_in_sync
        optimizer_start=time.time()
        optimizer=keras.optimizers.Adam(learning_rate=0.0001)
        print("optimizers took {} seconds".format(time.time()-optimizer_start))
        pretrained_start=time.time()
        pretrained_classifier=keras.models.load_model(args.pretrained_classifier_path)
        print('loading pretrained optimizer took {} seconds'.format(time.time()-pretrained_start))

        #optimizer=tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        model_start=time.time()
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
        print('time elapsed for making model = {}'.format(time.time()-model_start))

    print("mirrored stuff took {} seconds".format(time.time()-start))
    dataset_list=yvae_creativity_get_dataset_train(batch_size=args.batch_size,
                                                   dataset_names=args.dataset_names,
                                                   image_dim=args.image_dim,
                                                   mirrored_strategy=mirrored_strategy)
    dataset_dict={}
    test_dataset_dict={} #emoty because we don't really care about test scores for just pretraining creativity
    trainer=VAE_Creativity_Trainer(y_vae_list, args.epochs,dataset_dict, test_dataset_dict, optimizer,
                                   dataset_list,log_dir,mirrored_strategy, args.kl_loss_scale,
                                   callbacks=[],
                                   start_epoch=start_epoch,
                                   global_batch_size=GLOBAL_BATCH_SIZE,
                                   pretrained_classifier=pretrained_classifier,
                                   creativity_lambda=args.creativity_lambda,
                                   n_classes=n_classes
                                   )
    callbacks=[
        YvaeImageGenerationCallback(trainer, test_dataset_dict, save_folder, 3,enable_style_transfer=False)
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