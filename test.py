import datasets.dataset_factory as df
import tensorflow as tf

dataset_dir = '/User/hufangquan/code/SSD-Tensorflow-master/data/'
dataset_name = 'pascalvoc_2007'

# dataset = df.get_dataset(dataset_name, 'train', dataset_dir, 32,10000)
# iterator = dataset.make_one_shot_iterator()
# image_batch, label_batch, coord_batch = iterator.get_next()
coords = [[[1.1,2.1,3.1,4.1],[1.2,2.2,3.2,4.2]]]

print('')

