import tensorflow as tf
import numpy as np
from datasets import load_dataset
import sys 
sys.path.append("data_utils")
from processing_utils import *

BUFFER_SIZE = 10
BATCH_SIZE = 1
IMG_WIDTH = 512
IMG_HEIGHT = 512
AUTOTUNE = tf.data.AUTOTUNE

def cycle_get_datasets_train(batch_size=BATCH_SIZE,unit_test=False, content_path="jlbaker361/flickr_humans_mini", style_path="jlbaker361/anime_faces_mini",image_dim=512):
    train_style,train_content=get_datasets_train(unit_test=unit_test, content_path=content_path, style_path=style_path,image_dim=image_dim)
    return train_style.shuffle(BUFFER_SIZE).batch(batch_size), train_content.shuffle(BUFFER_SIZE).batch(batch_size)

def cycle_get_datasets_test(batch_size=BATCH_SIZE,unit_test=False, content_path="jlbaker361/flickr_humans_mini", style_path="jlbaker361/anime_faces_mini",image_dim=512):
    test_style,test_content=get_datasets_test(unit_test=unit_test, content_path=content_path, style_path=style_path,image_dim=image_dim)
    return test_style.shuffle(BUFFER_SIZE).batch(batch_size), test_content.shuffle(BUFFER_SIZE).batch(batch_size)