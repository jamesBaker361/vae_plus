import sys
sys.path.append("data_utils")
import tensorflow as tf

from processing_utils import *

BATCH_SIZE=4

def adain_get_dataset_train(batch_size=BATCH_SIZE,unit_test=False, content_path="jlbaker361/flickr_humans_mini", style_path="jlbaker361/anime_faces_mini",image_dim=512,strategy=None):
    train_style,train_content=get_datasets_train(unit_test=unit_test, content_path=content_path, style_path=style_path,image_dim=image_dim)
    return tf.data.Dataset.zip((train_style, train_content)).shuffle(batch_size*10).batch(batch_size)

def adain_get_dataset_test(batch_size=BATCH_SIZE,unit_test=False, content_path="jlbaker361/flickr_humans_mini", style_path="jlbaker361/anime_faces_mini",image_dim=512,strategy=None):
    test_style,test_content=get_datasets_test(unit_test=unit_test, content_path=content_path, style_path=style_path,image_dim=image_dim)
    return tf.data.Dataset.zip((test_style, test_content)).shuffle(batch_size*10).batch(batch_size)