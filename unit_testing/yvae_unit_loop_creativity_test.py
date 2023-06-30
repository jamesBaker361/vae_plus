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



def objective_unit_test_creative_load(image_dim):
    args.dataset_names=["jlbaker361/flickr_humans_mini", "jlbaker361/anime_faces_mini"]
    args.epochs=5
    args.load=False
    args.unfreezing_epoch=2
    args.fine_tuning=True
    args.image_dim=image_dim
    input_shape=(image_dim, image_dim, 3)
    creativity_encoder=get_encoder(input_shape,args.latent_dim, use_residual=args.use_residual, use_bn=args.use_bn,use_gn=args.use_gn)
    save_path='/scratch/jlb638/unit_testing/creativity_pretrained/'
    os.makedirs(save_path, exist_ok=True)
    creativity_encoder.save(save_path+ENCODER_NAME)
    args.pretrained_creativity_path=save_path+ENCODER_NAME
    objective_unit(None,args)



if __name__=='__main__':
    for dim in [64,128]:
        objective_unit_test_creative_load(dim)
