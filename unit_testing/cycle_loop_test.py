import sys
sys.path.append("cyclegan")
sys.path.append('/home/jlb638/Desktop/vae_plus/')
print(sys.path)
import tensorflow as tf


import unittest

from unittest.mock import patch
from cycle_loop import *
from string_globals import *

class Loop_Test(unittest.TestCase):

    def objective_test(self, image_dim):
        args.test=True
        args.epochs=2
        args.image_dim=image_dim
        objective(None,args)

    def objective_saving_test(self):
        print('objective_saving_test')
        args.epochs=11
        args.save=True
        args.name="save_test"
        args.interval=1
        args.threshold=10
        objective(None,args)

    def objective_load_test(self):
        print('objective_load_test')
        args.epochs=15
        args.load=True
        args.name="save_test"
        args.save=False
        objective(None,args)


if __name__ == '__main__':
    test=Loop_Test()
    print('begin')
    for dim in 64,128, 512:
        test.objective_test(dim)
    test.objective_saving_test()
    test.objective_load_test()
    print('end')