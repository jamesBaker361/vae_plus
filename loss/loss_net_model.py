from tensorflow import keras
from tensorflow.keras import layers

def get_loss_net(input_shape):
    vgg19 = keras.applications.VGG19(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    vgg19.trainable = False
    layer_names = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1"]
    outputs = [vgg19.get_layer(name).output for name in layer_names]
    mini_vgg19 = keras.Model(vgg19.input, outputs)

    inputs = layers.Input(shape=input_shape)
    mini_vgg19_out = mini_vgg19(inputs)
    return keras.Model(inputs, mini_vgg19_out, name="loss_net")