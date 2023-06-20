import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
import numpy as np
from scipy.linalg import sqrtm

def calculate_mu_sig(input_shape,images):
    model = InceptionV3(include_top=False, pooling='avg', input_shape=input_shape)
    images=tf.keras.applications.inception_v3.preprocess_input(images, data_format=None)
    resize_shape=(input_shape[0], input_shape[1])
    images=tf.image.resize(images, resize_shape)
    act = model.predict(images)
    # calculate mean and covariance statistics
    mu, sigma = act.mean(axis=0), np.cov(act, rowvar=False)
    return mu, sigma


def calculate_fid(input_shape, mu1, sigma1,mu2, sigma2):
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