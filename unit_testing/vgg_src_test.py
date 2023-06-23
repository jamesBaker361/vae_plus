import sys
sys.path.append('evaluation')
sys.path.append('yvae')
from yvae_data_helper import *

from vgg_src import *

def calculate_content_features_test(input_shape=(64,64,3),batch_size=4):
    images=tf.random.normal((batch_size, *input_shape))
    content_features=calculate_content_features(images)
    print(tf.shape(content_features))

def calculate_content_loss_test(input_shape=(64,64,3),batch_size=4):
    dataset_names=["jlbaker361/flickr_humans_dim_128_0.5k" ,"jlbaker361/anime_faces_dim_128_0.5k"]
    dataset_dict=yvae_get_dataset_train(dataset_names=dataset_names, image_dim=input_shape[1], batch_size=batch_size)
    samples={name : next(iter(dataset)) for name,dataset in dataset_dict.items()}
    names=[n for n in samples.keys()]
    for x in range(len(names)):
        for y in range(x+1):
            name1=names[x]
            name2=names[y]
            images1=samples[name1]
            images2=samples[name2]
            features1=calculate_content_features(images1)
            features2=calculate_content_features(images2)
            content_loss=calculate_content_loss(features1, features2)
            print(name1, name2, content_loss)

if __name__ == '__main__':
    calculate_content_features_test()
    calculate_content_loss_test()