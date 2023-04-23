import sys
sys.path.append('adain')

from model import *

def get_encoder_test(input_shape):
    encoder=get_encoder(input_shape)
    print(encoder.input_shape)
    print(encoder.output_shape)
    print(":-)")

def get_decoder_test():
    decoder=get_decoder()
    print(decoder.input_shape)
    print(decoder.output_shape)
    print(":-)")

def get_decoder_load():
    decoder=get_decoder()
    tf.saved_model.save("../../../../../scratch/jlb638/vae_plus/test/adain/adain_decoder")
    new_decoder=get_decoder("../../../../../scratch/jlb638/vae_plus/test/adain/")


if __name__ == '__main__':
    for s in 64,128,512:
        shape=(s,s,3)
        get_encoder_test(shape)
    