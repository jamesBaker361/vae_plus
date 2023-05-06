from typing import Any
import matplotlib.pyplot as plt
import sys
sys.path.append("data_utils")
import tensorflow as tf
import json
from processing_utils import *

class YvaeImageGenerationCallback:
    def __init__(self, yvae_trainer,test_dataset_dict,image_output_dir,batch_size):
        self.yvae_trainer=yvae_trainer
        self.test_dataset_dict=test_dataset_dict
        self.image_output_dir=image_output_dir
        self.sample={
            key: next(iter(value)) for key,value in test_dataset_dict.items()
        }
        self.batch_size=batch_size

    def __call__(self, epoch):
        self.style_transfer(self.sample, '{}/test_{}.png'.format(self.image_output_dir, epoch))
        random_sample={
            key: next(iter(value)) for key,value in self.test_dataset_dict.items()
        }
        self.style_transfer(random_sample,'{}/random_test_{}.png'.format(self.image_output_dir, epoch))
        self.generate_images('{}/gen_{}.png'.format(self.image_output_dir, epoch))

    def style_transfer(self,data,path):
        n_vae=len(self.yvae_trainer.y_vae_list)
        n_datasets=len(data)
        data_values=[v for v in data.values()]
        #print('tf.shape(data_values)',tf.shape(data_values))
        fig, axes = plt.subplots(nrows=n_vae, ncols=n_datasets, figsize=(20, 20))
        for i in range(n_vae):
            yvae=self.yvae_trainer.y_vae_list[i]
            for j in range(n_datasets):
                reconstructed_image=yvae(data_values[j])[0]
                #print('tf.shape(reconstructed_image)',tf.shape(reconstructed_image))
                reconstructed_image=denormalize(reconstructed_image[0])
                #print('tf.shape(reconstructed_image)',tf.shape(reconstructed_image))
                ax = axes[i][j]
                ax.imshow(reconstructed_image)
        plt.savefig(path)
        plt.close()
        plt.clf()

    def generate_images(self,path):
        images=self.yvae_trainer.generate_images(self.batch_size)
        n_vae=len(self.yvae_trainer.y_vae_list)
        fig, axes = plt.subplots(nrows=n_vae, ncols=self.batch_size, figsize=(20, 20))
        for i in range(n_vae):
            for j in range(self.batch_size):
                ax=axes[i][j]
                img=denormalize(images[i][j])
                ax.imshow(img)
        plt.savefig(path)
        plt.close()
        plt.clf()

class YvaeSavingCallback:
    def __init__(self, yvae_trainer, save_model_folder,threshold,interval):
        self.yvae_trainer=yvae_trainer
        self.save_model_folder=save_model_folder
        self.threshold=threshold
        self.interval=interval

    def __call__(self, epoch):
        if epoch % self.interval ==0 and epoch>=self.threshold:
            tf.saved_model.save(self.yvae_trainer.encoder,self.save_model_folder+"encoder")
            for x in range(len(self.yvae_trainer.decoders)):
                tf.saved_model.save(self.yvae_trainer.decoders[x],self.save_model_folder+"decoder_{}".format(x))
            print('saved at location {} epoch {}'.format(self.save_model_folder, epoch))
            meta_data = {"epoch":epoch}
            json_object = json.dumps(meta_data, indent=4)
 
            # Writing to sample.json
            with open(self.save_model_folder+"meta_data.json", "w+") as outfile:
                outfile.write(json_object)