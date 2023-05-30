import sys
sys.path.append('yvae')


from yvae_trainer import *
from yvae_callbacks import *
from yvae_data_helper import *
from yvae_unit_loop import *

def objective_unit_test(image_dim):
    args.load=False
    args.save=False
    args.epochs=2
    args.dataset_names=["jlbaker361/flickr_humans_mini", "jlbaker361/anime_faces_mini"]
    args.batch_size=4
    args.name='unit_unit_testing_{}'.format(image_dim)
    args.image_dim=image_dim
    objective_unit(None,args)

def objective_unit_test_save(image_dim):
    args.load=False
    args.save=True
    args.epochs=3
    args.dataset_names=["jlbaker361/flickr_humans_mini", "jlbaker361/anime_faces_mini"]
    args.threshold=0
    args.interval=1
    args.batch_size=4
    args.name='unit_unit_testing_{}'.format(image_dim)
    args.image_dim=image_dim
    objective_unit(None,args)


def obective_unit_test_load(image_dim):
    objective_unit_test_save(image_dim)
    args.load=True
    args.save=False
    args.epochs=5
    objective_unit(None,args)



if __name__=='__main__':
    for dim in [64,128,256]:
        objective_unit_test(dim)
        objective_unit_test_save(dim)
        obective_unit_test_load(dim)
