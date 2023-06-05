import sys
import os
sys.path.append('yvae')

from yvae_trainer import *
from yvae_callbacks import *
from yvae_data_helper import *
from yvae_creativity_loop import *
from yvae_model import *

args.epochs=2
args.batch_size=4
args.dataset_names=["jlbaker361/flickr_humans_mini", "jlbaker361/anime_faces_mini"]

def objective_creativity_test(image_dim):
    args.load=False
    args.save=False
    args.name='creativity_unit_testing_{}'.format(image_dim)
    args.image_dim=image_dim
    objective(None, args)

def objective_creativity_save(image_dim):
    args.load=False
    args.save=True
    args.name='creativity_unit_testing_{}'.format(image_dim)
    args.image_dim=image_dim
    args.interval=1
    args.threshold=1
    objective(None, args)

def objective_creativity_load(image_dim):
    objective_creativity_save(image_dim)
    args.epochs=5
    args.load=True
    args.save=False
    objective(None, args)

if __name__ =='__main__':
    for image_dim in [64,128]:
        n_classes=len(args.dataset_names)
        input_shape=(image_dim, image_dim,3)
        resnet_classifier=get_resnet_classifier(input_shape, n_classes)
        resnet_path='./temp/resnet/'
        os.makedirs(resnet_path, exist_ok=True)
        resnet_classifier.save(resnet_path)
        args.pretrained_classifier_path=resnet_path
        objective_creativity_test(image_dim)
        objective_creativity_load(image_dim)
    print('all done uwuwuwu')