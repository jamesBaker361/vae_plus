import sys
sys.path.append('adain')

from adain_loop import *

def objective_test(image_dim):
    args.load=False
    args.save=False
    args.name='objective_test'
    args.content_path='jlbaker361/flickr_humans_mini'
    args.style_path='jlbaker361/anime_faces_mini'
    args.batch_size=4
    args.image_dim=image_dim
    args.epochs=2
    objective(None,args)

def objective_test_save(image_dim):
    args.load=False
    args.save=True
    args.name='objective_test'
    args.content_path='jlbaker361/flickr_humans_mini'
    args.style_path='jlbaker361/anime_faces_mini'
    args.batch_size=4
    args.image_dim=image_dim
    args.epochs=2
    args.interval=1
    args.threshold=0
    objective(None,args)

def objective_test_load(image_dim):
    args.load=True
    args.save=False
    args.name='objective_test'
    args.content_path='jlbaker361/flickr_humans_mini'
    args.style_path='jlbaker361/anime_faces_mini'
    args.batch_size=4
    args.image_dim=image_dim
    args.epochs=3
    objective(None,args)


if __name__ =='__main__':
    for image_dim in [64,256,512]:
        objective_test(image_dim)
        objective_test_save(image_dim)
        objective_test_load(image_dim)
    print("all done with unit tests :)")