import sys
sys.path.append('adain')

from adain_model import *

def get_encoder_test(input_shape):
    encoder=get_encoder(input_shape)
    print(encoder.input_shape)
    print(encoder.output_shape)
    print("encoder  :-)")

def get_decoder_test():
    decoder=get_decoder()
    print(decoder.input_shape)
    print(decoder.output_shape)
    print("decoder :-)")

def get_decoder_load():
    decoder=get_decoder()
    tf.saved_model.save(decoder,"../../../../../scratch/jlb638/vae_plus/test/adain/adain_decoder")
    new_decoder=get_decoder("../../../../../scratch/jlb638/vae_plus/test/adain/")

def reconstruct_image_test(input_shape):
    print('recons test with shape ',input_shape)
    content=tf.random.normal((1,*input_shape))
    style=tf.random.normal((1,*input_shape))
    encoder=get_encoder(input_shape)
    decoder=get_decoder()
    style_encoded = encoder(style)
    content_encoded = encoder(content)

    # Compute the AdaIN target feature maps.
    t = ada_in(style=style_encoded, content=content_encoded)

    # Generate the neural style transferred image.
    reconstructed_image = decoder(t)
    print('content.shape',content.shape)
    print('style.shape', style.shape)
    print('reconstructed_image.shape',reconstructed_image.shape)



if __name__ == '__main__':
    for s in 64,128,512:
        shape=(s,s,3)
        get_encoder_test(shape)
        reconstruct_image_test(shape)
    get_decoder_test()
    get_decoder_load()
    print("all done :)")