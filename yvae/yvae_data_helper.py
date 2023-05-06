import sys
sys.path.append("data_utils")
import tensorflow as tf

from processing_utils import *

BATCH_SIZE=8

def yvae_get_dataset_train(batch_size=BATCH_SIZE,unit_test=False, dataset_names=["jlbaker361/flickr_humans_mini","jlbaker361/anime_faces_mini"],image_dim=512,strategy=None):
    dataset_dict={
        name: get_single_dataset_train(unit_test, name, image_dim).shuffle(batch_size*10).batch(batch_size) for name in dataset_names
    }
    return dataset_dict

def yvae_get_dataset_test(batch_size=BATCH_SIZE,unit_test=False, dataset_names=["jlbaker361/flickr_humans_mini","jlbaker361/anime_faces_mini"],image_dim=512,strategy=None):
    dataset_dict={
        name: get_single_dataset_test(unit_test, name, image_dim).shuffle(batch_size*10).batch(batch_size) for name in dataset_names
    }
    return dataset_dict