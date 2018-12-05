from __future__ import division
import os
import time
from glob import glob
import cv2
import csv
import scipy.ndimage
import tensorflow as tf
import numpy as np
from net.utils import conv_unit, deconv_unit, res_unit, conv3d
from net.common import multibox_layer
from net import common, losses, bboxes
import net.negative_filtering as neg_filter

config = {}
config['image_shape'] = [512,512]
config['feat_shapes'] = [(512,512),(256,256),(128,128),(64,64),(32,32),(16,16),(8,8)]
config['anchor_sizes'] = [(20.48, 51.2),
                      (51.2, 133.12),
                      (133.12, 215.04),
                      (215.04, 296.96),
                      (296.96, 378.88),
                      (378.88, 460.8),
                      (460.8, 542.72)]
config['anchor_ratios'] = [[2, .5],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5],
                       [2, .5]]
config['anchor_offset'] = 0.5
config['normalizations'] = [20, -1, -1, -1, -1, -1, -1]
config['arm_channels'] = [512]*7
config['arm_anchor_steps'] = [8, 16, 32, 64, 128, 256, 512]
config['anchor_scaling'] = [0.1,0.1,0.2,0.2]

config['interm_layer_channel'] = 0
config['chanel'] = 1
config['crop_size'] = [128, 128, 128]
config['stride'] = 4
config['max_stride'] = 16
config['num_neg'] = 800
config['th_neg'] = 0.02
config['th_pos_train'] = 0.5
config['th_pos_val'] = 1
config['num_hard'] = 4
config['bound_size'] = 12
config['reso'] = 1
config['sizelim'] = 3.  # mm
config['sizelim2'] = 20
config['sizelim3'] = 30
config['aug_scale'] = True
config['r_rand_crop'] = 0.3
config['pad_value'] = 170
config['augtype'] = {'flip': True, 'swap': False,
                     'scale': True, 'rotate': False}
config['blacklist'] = ['868b024d9fa388b7ddab12ec1c06af38',
                       '990fbe3f0a1b53878669967b9afd1441', 'adc3bbc63d40f8761c59be10f1e504c3']
# import logging
# logging.basicConfig(filename='LOG/'+__name__+'.log',format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', 
#                 level=print,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')

class Refine_det(object):
    def __init__(self, num_classes):
        self.config = config
        self.num_classes = num_classes
        self.ARM_LAYERSPOOL_STRIDE_SIZE = 2
        self.POOL_KERNEL_SIZE = 2
        self.POOL_STRIDE_SIZE = 2
        self.FEATURENUM_FORW = [24, 32, 64, 64, 64]
        self.FEATURENUM_BACK = [64, 64, 128]
        self.NUM_BLOCKS_FORW = [2, 2, 3, 3]
        self.NUM_BLOCKS_BACK = [3, 3]
        self.ARM_LAYERS = ['block3_1', 'block3','block4', 'block5_1', 'block5', 'block6_1','block6']
        self.KEEP_PROB = 0.5
        self.scope = 'refine_det_resnet'
        self.reuse = None
        self.config['num_classes'] = num_classes
        self.config['no_annotation_label'] = 21

    def model_func(self, image, is_training, input_data_format='channels_last'):

        end_points = {}
        with tf.variable_scope(self.scope, 'ssd_160_resnet', [image], reuse=self.reuse):
            # block 1
            net = conv_unit(input=image, output_chn=24, kernel_size=3, stride=1,
                            is_training=is_training, name='convpre_1')
            end_points['block1'] = net  # [512,512,24]
            print('block1---',net.get_shape().as_list())

            # block 2
            net = conv_unit(input=net, output_chn=24, kernel_size=3, stride=1,
                            is_training=is_training, name='convpre_2')
            end_points['block2'] = net  # [512,512, 24]
            print('block2---',net.get_shape().as_list())

            # block 3
            net = res_unit(net, 24, 32, stride=1, is_training=is_training, name="top_down_1_0")
            end_points['block3_1'] = net
            net = res_unit(net, 32, 32, stride=1, is_training=is_training, name="top_down_1_1")
            end_points['block3_2'] = net
            net = tf.layers.max_pooling2d(
                net, pool_size=self.POOL_KERNEL_SIZE, strides=self.POOL_STRIDE_SIZE)
            end_points['block3'] = net  # [256,256, 32]
            print('block3---',net.get_shape().as_list())

            # block 4
            net = res_unit(net, 32, 64, stride=1, is_training=is_training, name="top_down_2_0")
            end_points['block4_1'] = net
            net = res_unit(net, 64, 64, stride=1, is_training=is_training, name="top_down_2_1")
            end_points['block4_2'] = net
            net = tf.layers.max_pooling2d(
                net, pool_size=self.POOL_KERNEL_SIZE, strides=self.POOL_STRIDE_SIZE)
            end_points['block4'] = net  # [128,128, 64]
            print('block4---',net.get_shape().as_list())

            # block 5
            net = res_unit(net, 64, 64, stride=1, is_training=is_training, name="top_down_3_0")
            net = res_unit(net, 64, 64, stride=2, is_training=is_training, name="top_down_3_1")
            end_points['block5_1'] = net
            net = res_unit(net, 64, 64, stride=1, is_training=is_training, name="top_down_3_2")
            end_points['block5_2'] = net
            net = tf.layers.max_pooling2d(
                net, pool_size=self.POOL_KERNEL_SIZE, strides=self.POOL_STRIDE_SIZE)
            end_points['block5'] = net  # [32,32, 64]
            print('block5---',net.get_shape().as_list())

            # block 6
            net = res_unit(net, 64, 64, stride=1, is_training=is_training, name="top_down_4_0")
            net = res_unit(net, 64, 64, stride=2, is_training=is_training, name="top_down_4_1")
            end_points['block6_1'] = net
            net = res_unit(net, 64, 64, stride=1, is_training=is_training, name="top_down_4_2")
            end_points['block6_2'] = net
            net = tf.layers.max_pooling2d(
                net, pool_size=self.POOL_KERNEL_SIZE, strides=self.POOL_STRIDE_SIZE)
            end_points['block6'] = net  # [8, 8, 64]
            print('block6---',net.get_shape().as_list())

            # # block 7
            # net = deconv_unit(input=net, output_chn=64,
            #                   is_training=is_training, name="bottom_up_1_0")
            # end_points['block7'] = net  # [16, 16, 16, 64]
            # print('block7---',net.get_shape().as_list())
        
        # sizes = [[.07, .1025], [.15,.2121], [.3, .3674], [.45, .5196], [.6, .6708], \
        #     [.75, .8216], [.9, .9721]]
        # ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
        #     [1,2,.5,3,1./3], [1,2,.5], [1,2,.5]]
        return end_points
    
    def get_prematched_anchors(self, image_shape, gtlabels, gtboxes):
        anchor_boxes = common.ssd_anchors_all_layers(image_shape,config['feat_shapes'],
                           config['anchor_sizes'], config['anchor_ratios'],
                           config['anchor_steps'], offset=0.5, dtype=np.float32)
        arm_anchor_labels, arm_anchor_loc, arm_anchor_scores = common.anchor_match(gtlabels, gtboxes, 
                                                    anchor_boxes, self.config, anchor_for='arm')
        return arm_anchor_labels, arm_anchor_loc, arm_anchor_scores

    def forward(self, end_points, ground_truths, targets):
        gtboxes, gtlabels = ground_truths
        arm_anchor_labels, arm_anchor_loc, arm_anchor_scores = targets
        self.from_layers = []
        for name in self.ARM_LAYERS:
            self.from_layers.append(end_points[name])
        # get output of ARM and ODM
        arm_loc_layers, arm_cls_layers, odm_loc_layers, odm_cls_layers = multibox_layer(
            config, self.from_layers, num_classes=self.num_classes, clip=False)
        # arm_loc, arm_cls = common.concat_preds(arm_loc_layers, arm_cls_layers, 'arm')
        # odm_loc, odm_cls = common.concat_preds(odm_loc_layers, odm_cls_layers, 'odm')

        anchor_boxes = common.get_anchors(config, self.from_layers)
        # arm_anchor_labels, arm_anchor_loc, arm_anchor_scores = common.anchor_match(gtlabels, gtboxes, 
        #                                             anchor_boxes, self.config, anchor_for='arm')
        refined_anchors = common.refine_anchor(anchor_boxes, arm_loc_layers)
        end_points['refined_anchors'] = refined_anchors
        # end_points['odm_loc'] = odm_loc
        # end_points['odm_cls'] = odm_cls
        odm_anchor_labels, odm_anchor_loc, odm_anchor_scores = \
                        common.anchor_match(gtlabels, gtboxes, refined_anchors, self.config, anchor_for='odm')
        # filtered_anchors = neg_filter.filter_out(arm_cls, odm_cls, odm_loc_target_mask)
        arm_cls_loss, arm_loc_loss = losses.arm_losses(arm_cls_layers,arm_loc_layers,arm_anchor_labels, arm_anchor_loc, arm_anchor_scores)
        odm_cls_loss, odm_loc_loss = losses.odm_losses(odm_cls_layers,odm_loc_layers,odm_anchor_labels, odm_anchor_loc, odm_anchor_scores)
        return tf.add_n([arm_cls_loss, arm_loc_loss, odm_cls_loss, odm_loc_loss]), end_points

    def detect_bboxes(self, predictions, localisations,
                        select_threshold=None, nms_threshold=0.5,
                        clipping_bbox=None, top_k=400, keep_top_k=200):
        """Get the detected bounding boxes from the SSD network output.
        """
        # Select top_k bboxes from predictions, and clip
        rscores, rbboxes = \
            common.tf_ssd_bboxes_select(predictions, localisations,
                                            select_threshold=select_threshold,
                                            num_classes=self.params.num_classes)
        rscores, rbboxes = \
            bboxes.bboxes_sort(rscores, rbboxes, top_k=top_k)
        # Apply NMS algorithm.
        rscores, rbboxes = \
            bboxes.bboxes_nms_batch(rscores, rbboxes,
                                 nms_threshold=nms_threshold,
                                 keep_top_k=keep_top_k)
        if clipping_bbox is not None:
            rbboxes = bboxes.bboxes_clip(clipping_bbox, rbboxes)
        return rscores, rbboxes


def get_model():
    net = Refine_det(num_classes=20)
    return net
