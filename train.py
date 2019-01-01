# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import division
from __future__ import print_function

import argparse, functools, itertools, os, six
from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

from net import model, bboxes
from tf_extended import utils_func, metrics, tensors
from config_training import configuration as config
from datasets import dataset_factory
from preprocess import preprocessing_factory
import tf_utils

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
tf.logging.set_verbosity(tf.logging.INFO)

def get_model_fn(num_gpus, variable_strategy, num_workers):
    """Returns a function that will build the resnet model."""

    def _patech_model_fn(features, labels, mode, params):
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        weight_decay = params.weight_decay
        momentum = params.momentum

        tower_features = features
        tower_labels = labels
        tower_losses = []
        tower_gradvars = []
        tower_preds = []

        data_format = params.data_format
        if not data_format:
            if num_gpus == 0:
                data_format = 'channels_last'
            else:
                data_format = 'channels_first'

        if num_gpus == 0:
            num_devices = 1
            device_type = 'cpu'
        else:
            num_devices = num_gpus
            device_type = 'gpu'

        # setup devices and get losses/grads
        for i in range(num_devices):
            worker_device = '/{}:{}'.format(device_type, i)
            if variable_strategy == 'CPU':
                device_setter = utils_func.local_device_setter(
                    worker_device=worker_device)
            elif variable_strategy == 'GPU':
                device_setter = utils_func.local_device_setter(
                    ps_device_type='gpu',
                    worker_device=worker_device,
                    ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                        num_gpus, tf.contrib.training.byte_size_load_fn))

            with tf.variable_scope('resnet', reuse=bool(i != 0)):
                with tf.name_scope('tower_%d' % i) as name_scope:
                    with tf.device(device_setter):
                        loss, gradvars, preds = _tower_fn(
                            is_training, weight_decay,
                            [tower_features['image'][i], tower_features['coord'][i],
                            tower_features['anchor_label'][i],tower_features['anchor_loc'][i],
                            tower_features['anchor_score'][i]],
                            tower_labels[i], data_format)
                        tower_losses.append(loss)
                        tower_gradvars.append(gradvars)
                        tower_preds.append(preds)
                        if i == 0:
                            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                                           name_scope)

        # Now compute global loss and gradients.
        gradvars = []
        with tf.name_scope('gradient_averaging'):
            all_grads = {}
            for grad, var in itertools.chain(*tower_gradvars):
                if grad is not None:
                    all_grads.setdefault(var, []).append(grad)
            for var, grads in six.iteritems(all_grads):
                # Average gradients on the same device as the variables
                # to which they apply.
                with tf.device(var.device):
                    if len(grads) == 1:
                        avg_grad = grads[0]
                    else:
                        avg_grad = tf.multiply(
                            tf.add_n(grads), 1. / len(grads))
                    gradvars.append((avg_grad, var))

        # Device that runs the ops to apply global gradient updates.
        consolidation_device = '/gpu:0' if variable_strategy == 'GPU' else '/cpu:0'
        with tf.device(consolidation_device):
            num_batches_per_epoch = config['num_examples_per_epoch_train'] // (
                params.train_batch_size)
            # learning rate change epoch and corresponding learning rate
            boundaries = [
                num_batches_per_epoch * x
                for x in np.array([300, 400, 600], dtype=np.int64)
            ]
            staged_lr = [params.learning_rate *
                         x for x in [1, 0.1, 0.01, 0.002]]

            learning_rate = tf.train.piecewise_constant(tf.train.get_global_step(),
                                                        boundaries, staged_lr)

            # tensors_to_log not work
            tf.identity(learning_rate, name='learning_rate')
            tf.summary.scalar('learning_rate', learning_rate)

            loss = tf.reduce_mean(tower_losses, name='loss')

            examples_sec_hook = utils_func.ExamplesPerSecondHook(
                params.train_batch_size, every_n_steps=10)

            tensors_to_log = {'learning_rate': learning_rate, 'loss': loss}

            logging_hook = tf.train.LoggingTensorHook(
                tensors=tensors_to_log, every_n_iter=100)

            train_hooks = [logging_hook, examples_sec_hook]

            optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate, momentum=momentum)

            if params.sync:
                optimizer = tf.train.SyncReplicasOptimizer(
                    optimizer, replicas_to_aggregate=num_workers)
                sync_replicas_hook = optimizer.make_session_run_hook(
                    params.is_chief)
                train_hooks.append(sync_replicas_hook)

            # Create single grouped train op
            train_op = [
                optimizer.apply_gradients(
                    gradvars, global_step=tf.train.get_global_step())
            ]
            train_op.extend(update_ops)
            train_op = tf.group(*train_op)
            pred_scores = list(zip(*[p['score'] for p in tower_preds]))
            pred_bboxes = list(zip(*[p['bbox'] for p in tower_preds]))
            predictions = {'score': [tf.concat(score, axis=0) for score in pred_scores],
                        #    'classes': tf.concat([p['classes'] for p in tower_preds], axis=0),
                           'bbox': [tf.concat(bbox, axis=0) for bbox in pred_bboxes],
                           }
            rscores = predictions['score']
            rbboxes = predictions['bbox']
            tf.Print(rscores[0],[rscores[0]],'rscores',first_n=10)
            b_glabels = tf.concat(labels,axis=0)
            b_gbboxes = tf.concat(features['coord'],axis=0)
            # ==========================================================================
            # Performing post-processing on CPU: loop-intensive, usually more efficient.
            # ==========================================================================
            with tf.device('/device:CPU:0'):
                # Detected objects from SSD output.
                rscores, rbboxes = bboxes.detect_bboxes(rscores, rbboxes, num_classes=params.num_class,
                                                   select_threshold=params.select_threshold,
                                                   nms_threshold=params.nms_threshold,
                                                   clipping_bbox=None,
                                                   top_k=params.select_top_k,
                                                   keep_top_k=params.keep_top_k)
                # Compute TP and FP statistics.
                b_gdifficults = tf.zeros_like(b_glabels)
                num_gbboxes, tp, fp, rscores = \
                    bboxes.bboxes_matching_batch(rscores.keys(), rscores, rbboxes,
                                          b_glabels, b_gbboxes, b_gdifficults,
                                          matching_threshold=params.matching_threshold)

            # =================================================================== #
            # Evaluation metrics.
            # =================================================================== #

            with tf.device('/device:CPU:0'):
                dict_metrics = {}
                # Add metrics to summaries and Print on screen.
                for name, metric in dict_metrics.items():
                    # summary_name = 'eval/%s' % name
                    summary_name = name
                    op = tf.summary.scalar(
                        summary_name, metric[0], collections=[])
                    tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

                # FP and TP metrics.
                tp_fp_metric = metrics.streaming_tp_fp_arrays(
                    num_gbboxes, tp, fp, rscores)
                for c in tp_fp_metric[0].keys():
                    dict_metrics['tp_fp_%s' % c] = (tp_fp_metric[0][c],
                                                    tp_fp_metric[1][c])

                # Add to summaries precision/recall values.
                aps_voc = {}
                for c in tp_fp_metric[0].keys():
                    # Precison and recall values.
                    prec, rec = metrics.precision_recall(*tp_fp_metric[0][c])
                    summary_name_pr = 'precision/%s' % c
                    op_pr = tf.summary.tensor_summary(
                        summary_name_pr, prec, collections=[])
                    tf.add_to_collection(tf.GraphKeys.SUMMARIES, op_pr)

                    summary_name_rec = 'recall/%s' % c
                    op_rec = tf.summary.tensor_summary(
                        summary_name_rec, rec, collections=[])
                    tf.add_to_collection(tf.GraphKeys.SUMMARIES, op_rec)

                    # Average precision VOC07.
                    v = metrics.average_precision_voc07(prec, rec)
                    summary_name = 'AP_VOC/%s' % c
                    op = tf.summary.scalar(summary_name, v, collections=[])
                    tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
                    aps_voc[c] = v

                # # Mean average precision VOC07.
                # summary_name = 'AP_VOC/mAP'
                # mAP = tf.add_n(list(aps_voc.values())) / len(aps_voc)
                # op = tf.summary.scalar(summary_name, mAP, collections=[])
                # op = tf.Print(op, [mAP], summary_name)
                # tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
        for k,v in six.iteritems(predictions):
            new = []
            for tensor in v:
                tshape = tensors.get_shape(tensor)
                tshape.pop(1)
                tshape.pop(2)
                tshape[1] = -1
                tensor = tf.reshape(tensor, tf.stack(tshape))
                new.append(tensor)
            # for tensor in v:
            #     print('tensor %s shape is'%(tensor.name),tensor.get_shape().as_list())
            predictions[k] = tf.concat(new,axis=1)
        
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            # eval_metric_ops=aps_voc,
            training_hooks=train_hooks)

    return _patech_model_fn


def _tower_fn(is_training, weight_decay, feature, label, data_format):
    '''
    used for training of each gpu
    '''   
    image, bbox, arm_anchor_label, arm_anchor_loc, arm_anchor_scores = feature
    net = model.get_model()  # unused config, getpbb
    end_points = net.model_func(image, is_training=is_training,
                            input_data_format='channels_last')
    tower_loss, end_points = net.forward(end_points, [bbox,label],[arm_anchor_label, arm_anchor_loc, arm_anchor_scores])

    model_params = tf.trainable_variables()
    reg_loss = weight_decay * tf.add_n( # regularization
        [tf.nn.l2_loss(v) for v in model_params])

    # regularization loss of convolution
    # reg_loss = tf.reduce_sum(tf.get_collection(
    #     tf.GraphKeys.REGULARIZATION_LOSSES))
    tower_loss += reg_loss
    tower_grad = tf.gradients(tower_loss, model_params)

    return tower_loss, zip(tower_grad, model_params), end_points


def filenames(subset, data_dir):
    """Return filenames for dataset."""
    if subset == 'train':
        return os.path.join(data_dir, 'train')
    else:
        return os.path.join(data_dir, 'val')


def input_fn(data_dir,
             subset,
             num_shards,
             batch_size,
             use_distortion_for_training=True):

    with tf.device('/cpu:0'):
        # use_distortion = subset == 'train' and use_distortion_for_training
        # to be reshaped, first two elements are float64
        dataset_name = 'pascalvoc_2007'

        dataset = dataset_factory.get_dataset(dataset_name, subset, data_dir,
                                batch_size, config['image_shape'],config['num_epoches'])
        
        iterator = dataset.make_one_shot_iterator()
        batch_shape = [1,1,1] + [len(model.config['feat_shapes'])]*3
        batch_pack = iterator.get_next()
        print("batch_pack:", [t.get_shape().as_list() for t in batch_pack])
        if num_shards <= 1:
            # image_batch, label_batch, coord_batch, arm_anchor_labels_batch, arm_anchor_loc_batch,\
            #     arm_anchor_scores_batch = tf_utils.reshape_list(batch_pack,batch_shape)
            image_batch, label_batch, coord_batch, arm_anchor_labels_batch, arm_anchor_loc_batch,\
                arm_anchor_scores_batch = tf.train.batch(batch_pack, batch_size=batch_size)
            batch_queue = slim.prefetch_queue.prefetch_queue(
                tf_utils.reshape_list([image_batch, label_batch, coord_batch, arm_anchor_labels_batch, 
                arm_anchor_loc_batch, arm_anchor_scores_batch])
            )
            image_batch, label_batch, coord_batch, arm_anchor_labels_batch, arm_anchor_loc_batch,\
                arm_anchor_scores_batch = tf_utils.reshape_list(batch_queue.dequeue(), batch_shape)
            # No GPU available or only 1 GPU.
            return {'image': [image_batch], 'coord': [coord_batch], 'anchor_label':[arm_anchor_labels_batch],
                    'anchor_loc':[arm_anchor_loc_batch],'anchor_score':[arm_anchor_scores_batch]}, [label_batch]

        # Note that passing num=batch_size is safe here, even though
        # dataset.batch(batch_size) can, in some cases, return fewer than batch_size
        # examples. This is because it does so only when repeating for a limited
        # number of epochs, but our dataset repeats forever.
        batch_unstacked = [tf.unstack(data,num=batch_size,axis=0) for data in batch_pack]
        all_shards = [[[] for i in range(num_shards)] for j in range(len(batch_unstacked))]

        for i in xrange(batch_size):
            idx = i % num_shards
            for j,shards in enumerate(all_shards):
                shards[idx].append(batch_unstacked[j][i])
        
        all_feature_shards = [[tf.parallel_stack(x) for x in shards] for shards in all_shards]
        image_shards, label_shards, coord_shards, anchor_label_shards, anchor_loc_shards, \
            anchor_score_shards = tf_utils.reshape_list(all_feature_shards,batch_shape)

        return {'image': image_shards, 'coord': coord_shards,'anchor_label':anchor_label_shards,
                'anchor_loc':anchor_loc_shards,'anchor_score':anchor_score_shards}, label_shards


def get_experiment_fn(data_dir,
                      num_gpus,
                      variable_strategy,
                      use_distortion_for_training=True):

    def _experiment_fn(run_config, hparams):
        """Returns an Experiment."""
        # Create estimator.
        train_input_fn = functools.partial( # fix some argument in inpt_fn,
                                            # use it by passing the rest into it
            input_fn,
            data_dir,
            subset='train',
            num_shards=num_gpus,
            batch_size=hparams.train_batch_size,
            use_distortion_for_training=use_distortion_for_training)

        eval_input_fn = functools.partial(
            input_fn,
            data_dir,
            subset='val',
            batch_size=hparams.eval_batch_size,
            num_shards=num_gpus)

        num_eval_examples = config['num_examples_per_epoch_eval']
        if num_eval_examples % hparams.eval_batch_size != 0:
            raise ValueError(
                'validation set size must be multiple of eval_batch_size')

        train_steps = hparams.train_steps
        eval_steps = num_eval_examples // hparams.eval_batch_size

        classifier = tf.estimator.Estimator(
            model_fn=get_model_fn(num_gpus, variable_strategy,
                                  run_config.num_worker_replicas or 1),
            config=run_config,
            params=hparams)

        # Create experiment.
        return tf.contrib.learn.Experiment(
            classifier,
            train_input_fn=train_input_fn,
            eval_input_fn=eval_input_fn,
            train_steps=train_steps,
            eval_steps=eval_steps)

    return _experiment_fn


def main(model_dir, data_dir, num_gpus, variable_strategy,
         use_distortion_for_training, log_device_placement, num_intra_threads,
         **hparams):
        # The env variable is on deprecation path, default is set to off.
    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    # Session config.
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=log_device_placement,
        intra_op_parallelism_threads=num_intra_threads,
        gpu_options=tf.GPUOptions(force_gpu_compatible=True))

    config = utils_func.RunConfig(
        session_config=sess_config, model_dir=model_dir)
    tf.contrib.learn.learn_runner.run(
        get_experiment_fn(data_dir, num_gpus, variable_strategy,
                          use_distortion_for_training),
        run_config=config,
        hparams=tf.contrib.training.HParams(
            is_chief=config.is_chief,
            **hparams))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a Single-shot detection network')
    parser.add_argument('--train-path', dest='train_path', help='train record to use',
                        default=os.path.join(os.getcwd(), 'data', 'train.tfrecord'), type=str)
    parser.add_argument(
        '--model-dir',
        type=str,
        default=config['model_path'],
        help='The directory where the model will be stored.')
    parser.add_argument(
        '--data-dir',
        type=str,
        default=config['data_dir'],
        help='dataset directory')
    parser.add_argument(
        '--variable-strategy',
        choices=['CPU', 'GPU'],
        type=str,
        default='CPU',
        help='Where to locate variable operations')
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=config['num_gpus'],
        help='The number of gpus used. Uses only CPU if set to 0.')
    parser.add_argument(
        '--train-steps',
        type=int,
        default=config['train_steps'],
        help='The number of steps to use for training.')
    parser.add_argument(
        '--train-batch-size',
        type=int,
        default=config['batch_size'],
        help='Batch size for training.')
    parser.add_argument(
        '--eval-batch-size',
        type=int,
        default=config['batch_size'],
        help='Batch size for validation.')
    parser.add_argument(
        '--momentum',
        type=float,
        default=config['momentum'],
        help='Momentum for MomentumOptimizer.')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=config['weight_decay'],
        help='Weight decay for convolutions.')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=config['learning_rate'],
        help="""\
        This is the inital learning rate value. The learning rate will decrease
        during training. For more details check the model_fn implementation in
        this file.\
        """)
    parser.add_argument(
        '--use-distortion-for-training',
        type=bool,
        default=True,
        help='If doing image distortion for training.')
    parser.add_argument(
        '--sync',
        action='store_true',
        default=False,
        help="""\
        If present when running in a distributed environment will run on sync mode.\
        """)
    parser.add_argument(
        '--num-intra-threads',
        type=int,
        default=0,
        help="""\
        Number of threads to use for intra-op parallelism. When training on CPU
        set to 0 to have the system pick the appropriate number or alternatively
        set it to the number of physical CPU cores.\
        """)
    parser.add_argument(
        '--num-inter-threads',
        type=int,
        default=0,
        help="""\
        Number of threads to use for inter-op parallelism. If set to 0, the
        system will pick an appropriate number.\
        """)
    parser.add_argument(
        '--data-format',
        type=str,
        default=None,
        help="""\
        If not set, the data format best for the training device is used.
        Allowed values: channels_first (NCHW) channels_last (NHWC).\
        """)
    parser.add_argument('--num-class', dest='num_class', type=int, default=21,
                        help='number of classes, including background')
    parser.add_argument('--num-example', dest='num_example', type=int, default=16551,
                        help='number of image examples')
    parser.add_argument('--class-names', dest='class_names', type=str,
                        default='nodule')
    parser.add_argument('--nms-threshold', dest='nms_threshold', type=float, default=0.45,
                        help='non-maximum suppression threshold')
    parser.add_argument('--matching-threshold', dest='matching_threshold', type=float, default=0.5,
                        help='non-maximum suppression threshold')
    parser.add_argument('--select-threshold', dest='select_threshold', type=float, default=None,
                        help='pre select threshold????')
    parser.add_argument('--select-top-k', dest='select_top_k', type=int, default=400,
                        help='select top k bounding boxes????')
    parser.add_argument('--keep-top-k', dest='keep_top_k', type=int, default=200,
                        help='keep top k bounding boxes????')
    parser.add_argument('--overlap', dest='overlap_thresh', type=float, default=0.5,
                        help='evaluation overlap threshold')
    parser.add_argument('--force', dest='force_nms', type=bool, default=False,
                        help='force non-maximum suppression on different class')
    parser.add_argument(
        '--log-device-placement',
        action='store_true',
        default=False,
        help='Whether to log device placement.')
    args = parser.parse_args()

    if args.num_gpus > 0:
        assert tf.test.is_gpu_available(), "Requested GPUs but none found."
    if args.num_gpus < 0:
        raise ValueError(
            'Invalid GPU count: \"--num-gpus\" must be 0 or a positive integer.')
    if args.num_gpus == 0 and args.variable_strategy == 'GPU':
        raise ValueError('num-gpus=0, CPU must be used as parameter server. Set'
                         '--variable-strategy=CPU.')
    if args.num_gpus != 0 and args.train_batch_size % args.num_gpus != 0:
        raise ValueError('--train-batch-size must be multiple of --num-gpus.')
    if args.num_gpus != 0 and args.eval_batch_size % args.num_gpus != 0:
        raise ValueError('--eval-batch-size must be multiple of --num-gpus.')
    # print(vars(args))
    main(**vars(args))
