import sys
sys.path.append('adain')

import matplotlib.pyplot as plt
from adain_data_helper import *

def adain_get_dataset_train_unit_test():
    train=adain_get_dataset_train(batch_size=8,unit_test=True,image_dim=128)
    (sample_style, sample_content)=next(iter(train))
    plt.figure()
    plt.subplot(221)
    plt.title('style')
    plt.imshow(sample_style[0] * 0.5 + 0.5)
    plt.subplot(223)
    plt.title('content')
    plt.imshow(sample_content[0] * 0.5 + 0.5)

    plt.savefig("adain_train_get_datasets.png")
    plt.clf()

def adain_get_dataset_test_unit_test():
    test=adain_get_dataset_test(batch_size=2,unit_test=True,image_dim=128)
    (sample_style, sample_content)=next(iter(test))
    plt.figure()
    plt.subplot(221)
    plt.title('style')
    plt.imshow(sample_style[0] * 0.5 + 0.5)
    plt.subplot(223)
    plt.title('content')
    plt.imshow(sample_content[0] * 0.5 + 0.5)

    plt.savefig("adain_test_get_datasets.png")
    plt.clf()

def adain_get_dataset_train_real():
    train=adain_get_dataset_train(batch_size=8,image_dim=128)
    (sample_style, sample_content)=next(iter(train))
    plt.figure()
    plt.subplot(221)
    plt.title('style')
    plt.imshow(sample_style[0] * 0.5 + 0.5)
    plt.subplot(223)
    plt.title('content')
    plt.imshow(sample_content[0] * 0.5 + 0.5)

    plt.savefig("adain_train_get_datasets_real.png")
    plt.clf()

def adain_get_dataset_test_real():
    test=adain_get_dataset_test(batch_size=8,image_dim=128)
    (sample_style, sample_content)=next(iter(test))
    plt.figure()
    plt.subplot(221)
    plt.title('style')
    plt.imshow(sample_style[0] * 0.5 + 0.5)
    plt.subplot(223)
    plt.title('content')
    plt.imshow(sample_content[0] * 0.5 + 0.5)

    plt.savefig("adain_test_get_datasets_real.png")
    plt.clf()
    
if __name__ == '__main__':
    adain_get_dataset_train_unit_test()
    #adain_get_dataset_test_unit_test()
    adain_get_dataset_test_real()
    adain_get_dataset_train_real()