import sys
sys.path.append('yvae')

import matplotlib.pyplot as plt
from yvae_data_helper import *

def yvae_get_dataset_train_test():
    dataset_dict=yvae_get_dataset_train(batch_size=4, image_dim=128)
    for name,dataset in dataset_dict.items():
        plt.figure()
        img=next(iter(dataset))[0]
        plt.imshow(img*.5+0.5)
        name = name[len('jlbaker361/'):]
        plt.title(name)
        plt.savefig('yvae_{}_train.png'.format(name))


def yvae_get_dataset_test_test():
    dataset_dict=yvae_get_dataset_test(batch_size=4, image_dim=128)
    for name,dataset in dataset_dict.items():
        plt.figure()
        img=next(iter(dataset))[0]
        plt.imshow(img*.5+0.5)
        name = name[len('jlbaker361/'):]
        plt.title(name)
        plt.savefig('yvae_{}_test.png'.format(name))

def yvae_get_labeled_dataset_train_test():
    batch_size=4
    dataset_names=["jlbaker361/flickr_humans_mini", "jlbaker361/anime_faces_mini"]
    dataset=yvae_get_labeled_dataset_train(batch_size=batch_size, dataset_names=dataset_names,image_dim=128)
    sample=next(iter(dataset))
    (imgs,labels)=sample
    print(tf.shape(imgs))
    print(tf.shape(labels))
    for z in range(batch_size):
        plt.figure()
        plt.imshow(imgs[z]*0.5+0.5)
        plt.title(str(labels[z].numpy()))
        plt.savefig('yvae_{}_label_train.png'.format(z))
        plt.clf()

def yvae_get_labeled_dataset_test_test():
    batch_size=1
    dataset_names=["jlbaker361/flickr_humans_mini", "jlbaker361/anime_faces_mini"]
    dataset=yvae_get_labeled_dataset_test(batch_size=batch_size, dataset_names=dataset_names,image_dim=128)
    sample=next(iter(dataset))
    (imgs,labels)=sample
    print(tf.shape(imgs))
    print(tf.shape(labels))
    for z in range(batch_size):
        plt.figure()
        plt.imshow(imgs[z]*0.5+0.5)
        plt.title(str(labels[z].numpy()))
        plt.savefig('yvae_{}_label_test.png'.format(z))
        plt.clf()

if __name__ == '__main__':
    #yvae_get_dataset_train_test()
    #yvae_get_dataset_test_test()
    #yvae_get_labeled_dataset_train_test()
    yvae_get_labeled_dataset_test_test()
    print("all done with tests :-)")