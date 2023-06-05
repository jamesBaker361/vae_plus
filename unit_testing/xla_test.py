import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import tensorflow as tf
tf.config.optimizer.set_jit(True)
print(tf.config.optimizer.get_jit())
print('all done')