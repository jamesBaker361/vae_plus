import sys
sys.path.append('yvae')
from datasets import load_dataset

import matplotlib.pyplot as plt
from yvae_data_helper import *

def load_all_datasets():
    for num in [10,20,30,40,50]:
        flickr="jlbaker361/flickr_humans_{}k".format(num)
        anime="jlbaker361/anime_faces_{}k".format(num)
        load_dataset(flickr, split="train",cache_dir="../../../../../scratch/jlb638/hf_cache")
        load_dataset(anime, split="train",cache_dir="../../../../../scratch/jlb638/hf_cache")


def prod_dataset_test(image_dim=128):
    print('image dim ', image_dim)
    batch_size=2
    dataset_names=["jlbaker361/flickr_humans_10k" ,"jlbaker361/anime_faces_10k", "jlbaker361/artfaces_padded"]
    yvae_creativity_get_dataset_train(batch_size=batch_size,dataset_names=dataset_names,image_dim=image_dim,mirrored_strategy=None)
    yvae_get_labeled_dataset_train(batch_size=batch_size, dataset_names=dataset_names,image_dim=image_dim)
    yvae_get_labeled_dataset_train(batch_size=batch_size, dataset_names=dataset_names, image_dim=image_dim)
    dataset_names_medium=["jlbaker361/flickr_humans_20k" ,"jlbaker361/anime_faces_20k", "jlbaker361/artfaces_padded"]
    yvae_creativity_get_dataset_train(batch_size=batch_size,dataset_names=dataset_names_medium,image_dim=image_dim,mirrored_strategy=None)
    yvae_get_labeled_dataset_train(batch_size=batch_size, dataset_names=dataset_names_medium,image_dim=image_dim)
    yvae_get_labeled_dataset_train(batch_size=batch_size, dataset_names=dataset_names_medium, image_dim=image_dim)
    dataset_names_big=["jlbaker361/flickr_humans" ,"jlbaker361/anime_faces_50k", "jlbaker361/artfaces_padded"]
    yvae_creativity_get_dataset_train(batch_size=batch_size,dataset_names=dataset_names_big,image_dim=image_dim,mirrored_strategy=None)
    yvae_get_labeled_dataset_train(batch_size=batch_size, dataset_names=dataset_names_big,image_dim=image_dim)
    yvae_get_labeled_dataset_train(batch_size=batch_size, dataset_names=dataset_names_big, image_dim=image_dim)

if __name__=='__main__':
    load_all_datasets()
    exit()
    for dim in [128,256, 512]:
        prod_dataset_test(dim)