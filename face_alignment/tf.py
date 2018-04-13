import tensorflow as tf
import models_tensorflow

x = tf.placeholder(tf.float32, (1,128,128,3))
a = models_tensorflow.FAN(x)
print(a)