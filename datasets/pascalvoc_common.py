# Copyright 2015 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the Pascal VOC Dataset (images + annotations).
"""
import os, glob, functools

import tensorflow as tf
from datasets import dataset_utils
from preprocess import ssd_vgg_preprocessing
from net import model
import tf_utils

VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}
def _parser_fn(record, split_name, img_shape):
    # Features in Pascal VOC TFRecords.
    # refer to slim.tfexample_decoder.BoundingBox
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),
    }
    features = tf.parse_single_example(record, keys_to_features)

    # image = tf.decode_(features['image/encoded'], tf.float32)
    # image = tf.reshape(image, img_shape)
    image = tf.image.decode_jpeg(features['image/encoded'],channels=3)
    xmin = features['image/object/bbox/xmin'].values # since the original is tf sparse tensor, .value convert to Tensor
    ymin = features['image/object/bbox/ymin'].values
    xmax = features['image/object/bbox/xmax'].values
    ymax = features['image/object/bbox/ymax'].values
    bboxes = tf.stack([xmin,ymin,xmax,ymax],axis=-1)
    labels = features['image/object/bbox/label'].values
    if split_name == 'train':
        image, labels, bboxes = ssd_vgg_preprocessing.preprocess_for_train(image,labels,bboxes,img_shape)
    else:
        image, labels, bboxes = ssd_vgg_preprocessing.preprocess_for_eval(image, labels, bboxes,img_shape)
    net = model.get_model()
    arm_anchor_labels, arm_anchor_loc, arm_anchor_scores = net.get_prematched_anchors(img_shape,labels,bboxes)
    return tf_utils.reshape_list(image, labels, bboxes, arm_anchor_labels, arm_anchor_loc, arm_anchor_scores)

def get_split(split_name, dataset_dir, batch_size, image_shape, num_epoches, file_pattern,
              split_to_sizes, items_to_descriptions, num_classes):
    """Gets a dataset tuple with instructions for reading Pascal VOC dataset.

    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
        ValueError: if `split_name` is not a valid train/test split.
    """
    if split_name not in split_to_sizes:
        raise ValueError('split name %s was not recognized.' % split_name)
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)
    print(file_pattern, split_name)
    tf_filenames = glob.glob(file_pattern)
    dataset = tf.data.TFRecordDataset(tf_filenames)
    parser_fn = functools.partial(_parser_fn,
                        split_name=split_name, img_shape=image_shape)
    dataset = dataset.map(parser_fn)
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.repeat(num_epoches)
    # dataset = dataset.shuffle(configuration['shuffle_buff_size'])
    dataset = dataset.batch(batch_size)

    return dataset
