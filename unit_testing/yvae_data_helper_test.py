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

if __name__ == '__main__':
    yvae_get_dataset_train_test()
    yvae_get_dataset_test_test()