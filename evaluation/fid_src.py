import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
import numpy as np
from scipy.linalg import sqrtm


def calculate_fid(input_shape,images1,images2):
    # calculate activations
    model = InceptionV3(include_top=False, pooling='avg', input_shape=input_shape)
    images1=tf.keras.applications.inception_v3.preprocess_input(images1, data_format=None)
    images2=tf.keras.applications.inception_v3.preprocess_input(images2, data_format=None)
    resize_shape=(input_shape[0], input_shape[1])
    images1=tf.image.resize(images1, resize_shape)
    images2=tf.image.resize(images2, resize_shape)
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid