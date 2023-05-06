import sys
sys.path.append("cyclegan")
import tensorflow as tf


import unittest

from unittest.mock import patch

from cycle_data_helper import *
import matplotlib.pyplot as plt

class Cycle_Data_Loader_Test(unittest.TestCase):

    def train_get_datasets(self):
        train_style, train_content=cycle_get_datasets_train()
        random_jitter=get_random_jitter(512)
        sample_style = next(iter(train_style))
        sample_content=next(iter(train_content))
        plt.figure()
        plt.subplot(221)
        plt.title('style')
        plt.imshow(sample_style[0] * 0.5 + 0.5)

        plt.subplot(222)
        plt.title('style with random jitter')
        plt.imshow(random_jitter(sample_style[0]) * 0.5 + 0.5)

        plt.subplot(223)
        plt.title('content')
        plt.imshow(sample_content[0] * 0.5 + 0.5)

        plt.subplot(224)
        plt.title('content with random jitter')
        plt.imshow(random_jitter(sample_content[0]) * 0.5 + 0.5)

        plt.savefig("cyclegan_train_get_datasets.png")
        plt.clf()

    def test_get_datasets(self):
        train_style, train_content=cycle_get_datasets_test()
        random_jitter=get_random_jitter(512)
        sample_style = next(iter(train_style))
        sample_content=next(iter(train_content))
        plt.figure()
        plt.subplot(221)
        plt.title('style')
        plt.imshow(sample_style[0] * 0.5 + 0.5)

        plt.subplot(222)
        plt.title('style with random jitter')
        plt.imshow(random_jitter(sample_style[0]) * 0.5 + 0.5)

        plt.subplot(223)
        plt.title('content')
        plt.imshow(sample_content[0] * 0.5 + 0.5)

        plt.subplot(224)
        plt.title('content with random jitter')
        plt.imshow(random_jitter(sample_content[0]) * 0.5 + 0.5)

        plt.savefig("cyclegan_test_get_datasets.png")
        plt.clf()



if __name__ =="__main__":
    cdl_test=Cycle_Data_Loader_Test()
    cdl_test.train_get_datasets()
    cdl_test.test_get_datasets()
    print('all_done :)')