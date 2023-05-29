from processing_utils import *

path_list=["jlbaker361/flickr_humans_10k","jlbaker361/anime_faces_10k"]
dataset=get_labeled_datasets_generator_train(image_dim=64,path_list=path_list)

print(next(iter(dataset)))