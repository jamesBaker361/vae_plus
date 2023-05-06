import sys
sys.path.append("loss")
import tensorflow as tf

from loss_net_model import *

def get_loss_net_test(input_shape):
    loss_net= get_loss_net(input_shape)
    loss_net(tf.random.normal((4,*input_shape)))
    print('loss_net.input_shape',loss_net.input_shape)
    print('loss_net.output_shape',loss_net.output_shape)

if __name__ == '__main__':
    for dim in [64,128,512]:
        input_shape=(dim,dim,3)
        get_loss_net_test(input_shape)