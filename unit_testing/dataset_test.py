import sys
sys.path.append('yvae')
from datasets import load_dataset

import matplotlib.pyplot as plt
from yvae_data_helper import *

def load_all_datasets():
    for num in [0.5,5,10,20,30,40,50]:
        flickr="jlbaker361/flickr_humans_{}k".format(num)
        anime="jlbaker361/anime_faces_{}k".format(num)
        flickr_128="jlbaker361/flickr_humans_dim_128_{}k".format(num)
        anime_128="jlbaker361/anime_faces_dim_128_{}k".format(num)
        female_128='jlbaker361/kaggle_females_dim_128_{}k'.format(num)
        female_128='jlbaker361/kaggle_males_dim_128_{}k'.format(num)
        for name in [flickr,flickr_128,anime, anime_128,female_128]:
            load_dataset(name, split="train",cache_dir="../../../../../scratch/jlb638/hf_cache")


def prod_dataset_test(image_dim=128):
    print('image dim ', image_dim)
    batch_size=2
    dataset_names_tiny=["jlbaker361/flickr_humans_5k" ,"jlbaker361/anime_faces_5k", "jlbaker361/artfaces_padded_32"]
    dataset_names=["jlbaker361/flickr_humans_10k" ,"jlbaker361/anime_faces_10k", "jlbaker361/artfaces_padded"]
    dataset_names_medium=["jlbaker361/flickr_humans_20k" ,"jlbaker361/anime_faces_20k", "jlbaker361/artfaces_padded"]
    dataset_names_big=["jlbaker361/flickr_humans" ,"jlbaker361/anime_faces_50k", "jlbaker361/artfaces_padded"]
    for names in [dataset_names, dataset_names_medium, dataset_names_big, dataset_names_tiny]:
        yvae_creativity_get_dataset_train(batch_size=batch_size,dataset_names=names,image_dim=image_dim,mirrored_strategy=None)
        yvae_get_labeled_dataset_train(batch_size=batch_size, dataset_names=names,image_dim=image_dim)
        yvae_get_labeled_dataset_train(batch_size=batch_size, dataset_names=names, image_dim=image_dim)

if __name__=='__main__':
    load_all_datasets()
    for dim in [128,256, 512]:
        prod_dataset_test(dim)