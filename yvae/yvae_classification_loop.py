import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices --tf_xla_cpu_global_jit'
os.environ["XLA_FLAGS"] ="--xla_gpu_cuda_data_dir=/home/jlb638/.conda/envs/fine-tune/lib"
import tensorflow as tf
tf.config.optimizer.set_jit(True)
#from tensorflow_examples.models.pix2pix import pix2pix
from yvae_data_helper import *
from yvae_model import *
from yvae_trainer import *
from yvae_callbacks import *
from yvae_classification_trainer import *
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
parser.add_argument("--log_dir_parent",type=str,default="logs/")
parser.add_argument("--resnet",type=bool, default=False)
parser.add_argument("--external_name",type=str,default="",help='if set, whether to use external pretrained model')
parser.add_argument("--unfreezing_epoch",type=int,default=10,help='epoch at which to unfreeze pretrained external model')

args = parser.parse_args()

def objective(trial, args):
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

    mirrored_strategy = tf.distribute.MirroredStrategy(logical_gpus, cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

    if args.resnet:
        model_name=RESNET_CLASSIFIER
    else:
        model_name=CLASSIFIER_MODEL

    if args.external_name in set(EXTERNAL_NAME_LIST):
        use_external=True
    else:
        use_external=False


    with mirrored_strategy.scope():
        optimizer=keras.optimizers.Adam(learning_rate=0.0001)
        unfrozen_optimizer=keras.optimizers.Adam(learning_rate=0.00001)
        #optimizer=tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        if args.load:
            classifier_model=tf.keras.models.load_model(save_model_folder+model_name)
            with open(save_model_folder+"/meta_data.json","r") as src_file:
                start_epoch=json.load(src_file)["epoch"]

            print("successfully loaded from {} at epoch {}".format(save_model_folder, start_epoch),flush=True)
        else:
            if args.resnet:
                classifier_model=get_resnet_classifier(input_shape, n_classes)
            elif use_external:
                print('using external model')
                classifier_model=get_external_classifier(input_shape,args.external_name,n_classes)
            else:
                classifier_model = get_classifier_model(args.latent_dim,input_shape,n_classes)
    
    dataset=yvae_get_labeled_dataset_train(batch_size=args.batch_size, dataset_names=args.dataset_names,image_dim=args.image_dim)
    test_dataset=yvae_get_labeled_dataset_test(batch_size=args.batch_size, dataset_names=args.dataset_names,image_dim=args.image_dim)
    trainer=YVAE_Classifier_Trainer(classifier_model, args.epochs,optimizer, dataset, test_dataset=test_dataset,log_dir=log_dir,mirrored_strategy=mirrored_strategy,start_epoch=start_epoch,
                                    use_external=use_external,
                                    unfreezing_epoch=args.unfreezing_epoch,
                                    unfrozen_optimizer=unfrozen_optimizer)
    if args.save:
        trainer.callbacks=[YvaeClassifierSavingCallback(trainer, save_model_folder, args.threshold, args.interval,model_name=model_name)]
    print("begin loop :O")
    trainer.train_loop()

    print("end loop :)")

if __name__ == '__main__':
    print("begin")
    objective(None, args)
    print("end")