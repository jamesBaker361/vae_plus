import tensorflow as tf
from PIL import Image
import numpy as np
import os

import sys
sys.path.append("cyclegan")
from img_helpers import *


class CycleGANArtist:
    def __init__(self, save_model_folder, output_img_dir, content_dataset, limit=5000):
        self.generator_g=tf.saved_model.load(save_model_folder+"generator_g")
        self.generator_f=tf.saved_model.load(save_model_folder+"generator_f")
        self.limit = limit
        self.output_img_dir=output_img_dir
        os.makedirs(output_img_dir, exist_ok=True)
        self.content_dataset=content_dataset

    def __call__(self):
        # Generator G translates X -> Y
        # Generator F translates Y -> X.
        # x = style
        # y = content
        for count,content_img in enumerate( self.content_dataset):
            if count > self.limit:
                break
            save_single_image(self.generator_f, content_img, self.output_img_dir+"/image_{}.png".format(count))

class YVaeArtist: #one of these for each yvae in yvae list
    def __init__(self, save_model_folder,output_img_dir,decoder_index, limit=5000):
        self.encoder=tf.saved_model.load(save_model_folder+"encoder")
        self.decoder=tf.saved_model.load(save_model_folder+"decoder_{}".format(decoder_index))
        self.model=tf.keras.Sequential(
            [self.encoder, self.decoder]
        )
        self.limit=limit
        self.output_img_dir=output_img_dir
        os.makedirs(output_img_dir, exist_ok=True)

    def __call__(self):
        # x = style
        # y = content
        for count,content_img in enumerate( self.content_dataset):
            if count > self.limit:
                break
            save_single_image(self.generator_f, content_img, self.output_img_dir+"/image_{}.png".format(count))