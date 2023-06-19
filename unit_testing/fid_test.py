import sys
sys.path.append('evaluation')
from fid_src import *
import tensorflow as tf

def calculate_fid_test(initial_shape=(64,64,3),input_shape=(128,128,3),batch_size=10):
    images1=tf.random.normal((batch_size,*initial_shape))
    images2=tf.random.normal((batch_size,*initial_shape))
    fid=calculate_fid(input_shape, images1, images2)
    print(fid)

if __name__ == '__main__':
    calculate_fid_test()