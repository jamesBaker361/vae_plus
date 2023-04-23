import tensorflow as tf
from PIL import Image
import numpy as np

class Artist:
    def __call__(self,content_dataset, style_dataset, *args, **kwargs):
        pass

class CycleGANArtist(Artist):
    def __init__(self, save_model_folder, output_img_dir, limit=5000):
        self.generator_g=tf.saved_model.load(save_model_folder+"generator_g")
        self.generator_f=tf.saved_model.load(save_model_folder+"generator_f")
        self.limit = limit

    def __call__(self, content_dataset, style_dataset, output_img_dir):
        # Generator G translates X -> Y
        # Generator F translates Y -> X.
        # x = style
        # y = content
        count = 0
        output_img_dir+='/cycleGAN/'
        for content_img in content_dataset:
            stylized_img=self.generator_f(stylized_img)
            for img in stylized_img:
                img=np.array(img)
                Image.fromarray((img * 255).astype(np.uint8)).save(output_img_dir+'a-{}.png'.format(count))
                Image.fromarray((img * 0.5 + 0.5).astype(np.uint8)).save(output_img_dir+'b-{}.png'.format(count))
                Image.fromarray((img * 127.5 + 127.5).astype(np.uint8)).save(output_img_dir+'c-{}.png'.format(count))
                Image.fromarray((img).astype(np.uint8)).save(output_img_dir+'d-{}.png'.format(count))