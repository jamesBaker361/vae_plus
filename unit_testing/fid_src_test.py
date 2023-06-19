import sys
sys.path.append('evaluation')
sys.path.append('yvae')
from yvae_data_helper import *
from fid_src import *
import tensorflow as tf

def calculate_fid_test(initial_shape=(64,64,3),input_shape=(128,128,3),batch_size=4):
    images1=tf.random.normal((batch_size,*initial_shape))
    images2=tf.random.normal((batch_size,*initial_shape))
    fid=calculate_fid(input_shape, images1, images2)
    print(fid)

def  calculate_fid_test_real(initial_shape=(64,64,3),input_shape=(128,128,3),batch_size=4):
    dataset_names=["jlbaker361/flickr_humans_0.5k" ,"jlbaker361/anime_faces_0.5k"]
    dataset_dict=yvae_get_dataset_train(dataset_names=dataset_names, image_dim=initial_shape[1], batch_size=batch_size)
    samples={name : next(iter(dataset)) for name,dataset in dataset_dict.items()}
    names=[n for n in samples.keys()]
    for x in range(len(names)):
        for y in range(x+1):
            name1=names[x]
            name2=names[y]
            images1=samples[name1]
            images2=samples[name2]
            fid=calculate_fid(input_shape, images1, images2)
            print(name1, name2, fid)


if __name__ == '__main__':
    calculate_fid_test()
    calculate_fid_test_real()
    print(' all done yay')