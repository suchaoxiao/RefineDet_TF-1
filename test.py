import datasets.dataset_factory as df

dataset_dir = '/User/hufangquan/code/SSD-Tensorflow-master/data/'
dataset_name = 'pascalvoc_2007'

dataset = df.get_dataset(dataset_name, 'train', dataset_dir, 32,10000)
iterator = dataset.make_one_shot_iterator()
image_batch, label_batch, coord_batch = iterator.get_next()
print('')

