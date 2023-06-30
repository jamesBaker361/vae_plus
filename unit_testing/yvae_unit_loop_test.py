import sys
sys.path.append('yvae')
import os
sys.path.append(os.getcwd())
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["XLA_FLAGS"] ="--xla_gpu_cuda_data_dir=/home/jlb638/.conda/envs/fine-tune/lib"
import tensorflow as tf
tf.config.optimizer.set_jit(True)


from yvae.yvae_trainer import *
from yvae.yvae_callbacks import *
from yvae.yvae_data_helper import *
from yvae.yvae_unit_loop import *
from yvae.yvae_model import *

def objective_unit_test(image_dim):
    args.load=False
    args.save=False
    args.epochs=5
    args.dataset_names=["jlbaker361/flickr_humans_mini", "jlbaker361/anime_faces_mini"]
    args.batch_size=4
    args.name='unit_unit_testing_dont_save_{}'.format(image_dim)
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
        args.use_residual=True
        objective_unit_test(dim)
        objective_unit_test_save(dim)
        obective_unit_test_load(dim)
        args.use_gn=True
        objective_unit_test(dim)
        objective_unit_test_save(dim)
        obective_unit_test_load(dim)
