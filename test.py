import datasets.dataset_factory as df
import tensorflow as tf
import numpy as np
tf.enable_eager_execution()
dataset_dir = '/User/hufangquan/code/SSD-Tensorflow-master/data/'
dataset_name = 'pascalvoc_2007'

# dataset = df.get_dataset(dataset_name, 'train', dataset_dir, 32,10000)
# iterator = dataset.make_one_shot_iterator()
# image_batch, label_batch, coord_batch = iterator.get_next()
a = tf.ones([1,32,32,4])
b = tf.ones([None,1,1,1])
a = tf.maximum(a,b)


print('')

