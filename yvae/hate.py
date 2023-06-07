import sys
sys.path.append("../data_utils")
from yvae_data_helper import *

dataset_names=["jlbaker361/flickr_humans_10k","jlbaker361/anime_faces_10k"]
test_dataset=yvae_get_labeled_dataset_test(4, dataset_names=dataset_names,image_dim=64)

for t in test_dataset:
    print(t)