import sys
sys.path.append("cyclegan")
import tensorflow as tf

from cycle_model import *

def unet_generator_test(image_dim):
    output_channels=3
    norm_type='instancenorm'
    noise=tf.random.normal((1,image_dim,image_dim,output_channels))
    unet=unet_generator(output_channels, image_dim, norm_type)
    print(tf.shape(noise), tf.shape(unet(noise)))

def discriminator_test(image_dim):
    output_channels=3
    norm_type='instancenorm'
    disc=get_discriminator(norm_type, False)
    noise=tf.random.normal((1,image_dim,image_dim,output_channels))
    print(tf.shape(noise), tf.shape(disc(noise)))


if __name__=='__main__':
    for dim in 64,128,256,512:
        print('============== {} ==========='.format(dim))
        unet_generator_test(dim)
        discriminator_test(dim)