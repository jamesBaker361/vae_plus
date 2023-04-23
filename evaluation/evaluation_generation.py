from artists import *

class ImageGeneration:
    def __init__(self, artists,content_dataset, style_dataset,output_img_dir):
        self.artists=artists
        self.content_dataset=content_dataset
        self.style_dataset=style_dataset
        self.output_img_dir=output_img_dir

    def __call__(self,*args, **kwds):
        for artist in self.artists:
            artist(self.content_dataset, self.style_dataset, self.output_img_dir)