import sys
sys.path.append('yvae')


from yvae_trainer import *
from yvae_callbacks import *
from yvae_data_helper import *
from yvae_classification_loop import *

def objective_test(image_dim):
    args.load=False
    args.save=False
    args.epochs=12
    args.dataset_names=["jlbaker361/flickr_humans_mini", "jlbaker361/anime_faces_mini"]
    args.image_dim=image_dim
    objective(None,args)

def objective_test_save(image_dim):
    args.load=False
    args.save=True
    args.epochs=2
    args.dataset_names=["jlbaker361/flickr_humans_mini", "jlbaker361/anime_faces_mini"]
    args.image_dim=image_dim
    args.save=True
    args.interval=1
    args.threshold=1
    objective(None,args)

def objective_test_load(image_dim):
    args.load=True
    args.epochs=5
    args.dataset_names=["jlbaker361/flickr_humans_mini", "jlbaker361/anime_faces_mini"]
    args.image_dim=image_dim
    objective(None,args)

def objective_external_test(image_dim):
    args.load=False
    args.save=False
    args.epochs=12
    args.dataset_names=["jlbaker361/flickr_humans_mini", "jlbaker361/anime_faces_mini"]
    args.image_dim=image_dim
    objective(None,args)

if __name__=='__main__':
    for dim in [32,128]: #,128,512]:
        args.name='external_unit_testing_{}'.format(dim)
        args.external_name=VGG
        args.unfreezing_epoch=4
        args.resnet=False
        objective_test(dim)
        objective_test_save(dim)
        objective_test_load(dim)
    print("all done :)))")
