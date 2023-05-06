from typing import Any
from adain_model import *
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("data_utils")
import tensorflow as tf

from processing_utils import *



class AdainImageGenerationCallback:
    def __init__(self,adain_trainer, test_dataset,image_output_dir):
        self.adain_trainer=adain_trainer
        self.sample=next(iter(test_dataset))
        self.test_dataset=test_dataset
        self.image_ouput_dir=image_output_dir

    def get_reconstructed_image(self, style, content):
        style_encoded = self.adain_trainer.encoder(style)
        content_encoded = self.adain_trainer.encoder(content)

        # Compute the AdaIN target feature maps.
        t = ada_in(style=style_encoded, content=content_encoded)

        # Generate the neural style transferred image.
        reconstructed_image = self.adain_trainer.decoder(t)
        return reconstructed_image

    def __call__(self, epoch):
        style, content = self.sample
        reconstructed_image = self.get_reconstructed_image(style, content)
        self.save_fig(style,content, reconstructed_image, '{}/test_{}.png'.format(self.image_ouput_dir, epoch))

        (random_style, random_content) = next(iter(self.test_dataset))
        random_reconstructed_image=self.get_reconstructed_image(random_style, random_content)
        self.save_fig(random_style, random_content, random_reconstructed_image, '{}/random_test_{}.png'.format(self.image_ouput_dir, epoch))


    def save_fig(self, style, content, reconstructed_image, path):
        style=denormalize(style[0])
        content=denormalize(content[0])
        reconstructed_image=denormalize(reconstructed_image[0])
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

        # Plot the first image on the left subplot
        ax1.imshow(style)
        ax1.set_title('style')

        # Plot the second image on the right subplot
        ax2.imshow(content)
        ax2.set_title('content')

        ax3.imshow(reconstructed_image)
        ax3.set_title('reconstructed')

        # Set the spacing between the subplots
        fig.subplots_adjust(wspace=0.5)

        # Show the figure
        plt.savefig(path)
        plt.clf()
        plt.close('all')

class AdainModelSaveCallback:
    def __init__(self, adain_trainer):
        self.adain_trainer=adain_trainer

    def __call__(self,epoch):
        self.adain_trainer.save_decoder()