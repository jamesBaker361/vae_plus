import tensorflow as tf
#from tensorflow_examples.models.pix2pix import pix2pix
from yvae_data_helper import *
from yvae_model import *
from yvae_trainer import *
from yvae_callbacks import *
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
parser.add_argument("--save_img_parent",type=str,default="/home/jlb638/Desktop/vae_plus/gen_imgs/yvae/")
parser.add_argument("--name",type=str,default="cycle_{}".format(str(datetime.now(timezone.utc))))
parser.add_argument("--save_model_parent", type=str,default="../../../../../scratch/jlb638/yvae_models/yvae/")
parser.add_argument("--dataset_names",nargs="+",default=["jlbaker361/flickr_humans_10k", "jlbaker361/anime_faces_10k","jlbaker361/ar_rom_bar_ren" ])
parser.add_argument("--load", type=bool, default=False, help="whether to load previous model if possible")
parser.add_argument("--save", type=bool, default=False, help='whether to save model')
parser.add_argument("--image_dim",type=int, default=128)
parser.add_argument("--interval",type=int,default=10,help='save model every interval # of epochs')
parser.add_argument("--threshold",type=int,default=50,help='epoch threshold for when to start saving')
parser.add_argument("--latent_dim",type=int, default=32,help='latent dim for encoding')

args = parser.parse_args()

from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

def objective(trial,args):
    save_folder=args.save_img_parent+args.name+"/"
    save_model_folder=args.save_model_parent+args.name+"/"
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(save_model_folder, exist_ok=True)