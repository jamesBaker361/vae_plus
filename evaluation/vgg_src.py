import tensorflow as tf
from tensorflow.keras.applications import VGG19

def calculate_content_features(images, content_layer = 'block4_conv2'):
    # Load pre-trained VGG19 model
    vgg_model = VGG19(include_top=False, weights='imagenet')

    # Get the output of the VGG model at a specific layer
    vgg_content_model = tf.keras.Model(inputs=vgg_model.input, outputs=vgg_model.get_layer(content_layer).output)

    # Preprocess images
    images = tf.keras.applications.vgg19.preprocess_input(images)

    # Compute the feature representations
    content_features = vgg_content_model(images)

    return content_features

def calculate_content_loss(features1, features2):
    content_loss = tf.reduce_mean(tf.square(features1 - features2))

    return content_loss