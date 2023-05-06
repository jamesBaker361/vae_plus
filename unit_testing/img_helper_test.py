import sys
sys.path.append("cyclegan")
import tensorflow as tf

from img_helpers import *
sys.path.append('adain')

from adain_data_helper import *

def generate_single_image_test():
    test_data=adain_get_dataset_test(batch_size=1,image_dim=128)
    test_input=next(iter(test_data))[0]
    def get_model():
        def _model(img):
            return img
        return _model
    model=get_model()
    generate_single_image(model, test_input, "exploration/img_helper_test.png")

if __name__=='__main__':
    generate_single_image_test()