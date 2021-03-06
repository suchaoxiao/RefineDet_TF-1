from __future__ import division
import os
import time
from glob import glob
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
config['anchor_sizes'] = [(20.48, 51.2),  #前面的是小的正方形defaultbox或者prior box，后面是大的
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
# 定义refindet网络模型
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
    #定义网络功能
    def model_func(self, image, is_training, input_data_format='channels_last'):
        #endpoint用来存储临时状态
        end_points = {}
        with tf.variable_scope(self.scope, 'ssd_160_resnet', [image], reuse=self.reuse):
            # block 1  convunit 是cnn+bn+relu  默认padding=valid
            net = conv_unit(input=image, output_chn=24, kernel_size=3, stride=1,
                            is_training=is_training, name='convpre_1')
            end_points['block1'] = net  # [512,512,24]
            print('block1---',net.get_shape().as_list())

            # block 2
            net = conv_unit(input=net, output_chn=24, kernel_size=3, stride=1,
                            is_training=is_training, name='convpre_2')
            end_points['block2'] = net  # [512,512, 24]
            print('block2---',net.get_shape().as_list())

            # block 3  resnet网络的输出尺寸没有变化，就是shortcut+cnn
            net = res_unit(net, 24, 32, stride=1, is_training=is_training, name="top_down_1_0")
            end_points['block3_1'] = net
            net = res_unit(net, 32, 32, stride=1, is_training=is_training, name="top_down_1_1")
            end_points['block3_2'] = net
            #最大池化 poolsize=2 尺寸减半
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
            #下面的stride=2将图片缩小一半，[64,64,64]
            net = res_unit(net, 64, 64, stride=2, is_training=is_training, name="top_down_3_1")
            end_points['block5_1'] = net
            net = res_unit(net, 64, 64, stride=1, is_training=is_training, name="top_down_3_2")
            end_points['block5_2'] = net
            #maxpool  又缩小一半[32,32,64]
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
            tf.Print(net, [net], message='debuuuuuuuuuuuug', summarize=100)

            # # block 7
            # net = deconv_unit(input=net, output_chn=64,
            #                   is_training=is_training, name="bottom_up_1_0")
            # end_points['block7'] = net  # [16, 16, 16, 64]
            # print('block7---',net.get_shape().as_list())
        
        # sizes = [[.07, .1025], [.15,.2121], [.3, .3674], [.45, .5196], [.6, .6708], \
        #     [.75, .8216], [.9, .9721]]
        # ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
        #     [1,2,.5,3,1./3], [1,2,.5], [1,2,.5]]
        return end_points   #最后返回网络输出  modelfun函数的返回
    #定义获取提前匹配的anchor label 坐标还有logit得分  label就是21个类别标签，score就是框的iou值
    def get_prematched_anchors(self, image_shape, gtlabels, gtboxes):
        #调用了ssd中产生anchor 的方法
        anchor_boxes = common.ssd_anchors_all_layers(image_shape,config['feat_shapes'],
                           config['anchor_sizes'], config['anchor_ratios'],
                           config['arm_anchor_steps'], offset=0.5, dtype=np.float32)
        #调用anchormatch函数计算anchor和gtbox匹配情况（类似rpn网络）
        arm_anchor_labels, arm_anchor_loc, arm_anchor_scores = common.anchor_match(gtlabels, gtboxes,
                                                    anchor_boxes, self.config, anchor_for='arm')
        return arm_anchor_labels, arm_anchor_loc, arm_anchor_scores
    #定义前传函数
    def forward(self, end_points, ground_truths, targets):
        #获取gtlabel和gtbox
        gtboxes, gtlabels = ground_truths
        #获取arm网络输出的anchor类别（fg/bg） anchor坐标和anchor logits得分
        arm_anchor_labels, arm_anchor_loc, arm_anchor_scores = targets
        self.from_layers = []
        for name in self.ARM_LAYERS:
            self.from_layers.append(end_points[name])
        # get output of ARM and ODM
        #调用multiboxlayer函数，
        arm_loc_layers, arm_cls_layers, odm_loc_layers, odm_cls_layers = multibox_layer(
            config, self.from_layers, num_classes=self.num_classes, clip=False)
        # arm_loc, arm_cls = common.concat_preds(arm_loc_layers, arm_cls_layers, 'arm')
        # odm_loc, odm_cls = common.concat_preds(odm_loc_layers, odm_cls_layers, 'odm')
        #获取anchor
        anchor_boxes = common.get_anchors(config, self.from_layers)
        # arm_anchor_labels, arm_anchor_loc, arm_anchor_scores = common.anchor_match(gtlabels, gtboxes, 
        #                                             anchor_boxes, self.config, anchor_for='arm')

        #对anchor box进行修正
        refined_anchors = common.refine_anchor(anchor_boxes, arm_loc_layers)
        end_points['refined_anchors'] = refined_anchors
        #对arm网络输出和gtbox信息送到odm网络进行匹配
        odm_anchor_labels, odm_anchor_loc, odm_anchor_scores = \
                        common.anchor_match(gtlabels, gtboxes, refined_anchors, self.config, anchor_for='odm')
        #####
        odm_scores = tf.Print(odm_anchor_scores[0],[odm_anchor_scores[0]],message='odm_scores',summarize=100)
        #####
        # make losses   获取loss=arm loss+odm loss
        arm_cls_loss, arm_loc_loss = losses.arm_losses(arm_cls_layers,arm_loc_layers,arm_anchor_labels, arm_anchor_loc, arm_anchor_scores)
        odm_cls_loss, odm_loc_loss = losses.odm_losses(odm_cls_layers,odm_loc_layers,odm_anchor_labels, odm_anchor_loc, odm_anchor_scores)
        end_points['score'] = odm_cls_layers
        end_points['bbox']  = odm_loc_layers
        return tf.add_n([arm_cls_loss, arm_loc_loss, odm_cls_loss, odm_loc_loss]), end_points


def get_model():
    net = Refine_det(num_classes=20)
    return net
