import sys
sys.path.append("data_utils")
from processing_utils import *
import tensorflow as tf

def equal_length_test():
    big_length=5
    smaller=[_ for _ in range(1)]
    bigger = [_ for _ in range(big_length)]
    smaller, bigger = equal_length(smaller, bigger)
    assert len(bigger) == len(smaller)
    assert big_length ==len(smaller)

def equal_length_switched_test():
    big_length=5
    smaller=[_ for _ in range(1)]
    bigger = [_ for _ in range(big_length)]
    smaller, bigger = equal_length(bigger, smaller)
    assert len(bigger) == len(smaller)
    assert big_length ==len(smaller)

def get_labeled_datasets_train_test():
    path_list=["jlbaker361/flickr_humans_mini", "jlbaker361/anime_faces_mini"]
    image_dim=128
    zipped_dataset=get_labeled_datasets_train(path_list, image_dim)
    (image, label)=next(iter(zipped_dataset))
    print(tf.shape(image))
    print(tf.shape(label))

if __name__=='__main__':
    equal_length_test()
    equal_length_switched_test()
    get_labeled_datasets_train_test()
    print("all done :-)")