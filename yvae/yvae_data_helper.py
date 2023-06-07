import sys
sys.path.append("data_utils")
import tensorflow as tf

from processing_utils import *

BATCH_SIZE=8

def yvae_get_labeled_dataset_train(batch_size=BATCH_SIZE,dataset_names=["jlbaker361/flickr_humans_mini","jlbaker361/anime_faces_mini"],image_dim=512,mirrored_strategy=None):
    dataset=get_labeled_datasets_generator_train(dataset_names, image_dim).shuffle(batch_size*100, seed=1234).batch(batch_size)
    if mirrored_strategy is not None:
        dataset=mirrored_strategy.experimental_distribute_dataset(dataset)
    return dataset

def yvae_get_labeled_dataset_test(batch_size=BATCH_SIZE,dataset_names=["jlbaker361/flickr_humans_mini","jlbaker361/anime_faces_mini"],image_dim=512,mirrored_strategy=None):
    dataset=get_labeled_datasets_generator_test(dataset_names, image_dim).shuffle(batch_size*100,seed=1234).batch(batch_size)
    if mirrored_strategy is not None:
        dataset=mirrored_strategy.experimental_distribute_dataset(dataset)
    return dataset

def yvae_get_dataset_train(batch_size=BATCH_SIZE,unit_test=False, dataset_names=["jlbaker361/flickr_humans_mini","jlbaker361/anime_faces_mini"],image_dim=512,mirrored_strategy=None):
    #Consider either turning off auto-sharding or switching the auto_shard_policy to DATA to shard this dataset. You can do this by creating a new `tf.data.Options()` object then setting `options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA` before applying the options object to the dataset via `dataset.with_options(options)`.
    options=tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    if mirrored_strategy is None:
        dataset_dict={
            name: get_single_dataset_train(unit_test, name, image_dim).shuffle(batch_size*100).batch(batch_size).with_options(options) for name in dataset_names
        }
    else:
        dataset_dict={
            name: mirrored_strategy.experimental_distribute_dataset(get_single_dataset_train(unit_test, name, image_dim).shuffle(batch_size*10).batch(batch_size)) for name in dataset_names
        }
    return dataset_dict

def yvae_get_dataset_test(batch_size=BATCH_SIZE,unit_test=False, dataset_names=["jlbaker361/flickr_humans_mini","jlbaker361/anime_faces_mini"],image_dim=512,mirrored_strategy=None):
    if mirrored_strategy is None:
        dataset_dict={
            name: get_single_dataset_test(unit_test, name, image_dim).shuffle(batch_size*100).batch(batch_size) for name in dataset_names
        }
    else:
        dataset_dict={
            name: mirrored_strategy.experimental_distribute_dataset(get_single_dataset_test(unit_test, name, image_dim).shuffle(batch_size*10).batch(batch_size)) for name in dataset_names
        }
    return dataset_dict

def yvae_creativity_get_dataset_train(batch_size=BATCH_SIZE,dataset_names=["jlbaker361/flickr_humans_mini","jlbaker361/anime_faces_mini"],image_dim=512,mirrored_strategy=None):
    dataset= get_multiple_datasets_train(dataset_names,image_dim,preprocess=False).shuffle(batch_size*100).batch(batch_size)
    if mirrored_strategy is not None:
        dataset=mirrored_strategy.experimental_distribute_dataset(dataset)
    return dataset

def yvae_creativity_get_dataset_test(batch_size=BATCH_SIZE,dataset_names=["jlbaker361/flickr_humans_mini","jlbaker361/anime_faces_mini"],image_dim=512,mirrored_strategy=None):
    dataset= get_multiple_datasets_test(dataset_names,image_dim).shuffle(batch_size*100).batch(batch_size)
    if mirrored_strategy is not None:
        dataset=mirrored_strategy.experimental_distribute_dataset(dataset)
    return dataset