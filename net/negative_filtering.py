# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


def filter_out(arm_cls_preds, odm_anchor_cls_label, odm_loc_target_mask):
    # anchor refine module: arm
    # object_detection_module: odm
    '''
    arm_cls_preds: (batch, 2, num_anchors) like tensor
                the foreground/background prediction output of arm layer
    odm_anchor_cls_label: (batch, num_anchors) like tensor
                odm class label
    odm_loc_target_mask: (batch, num_anchors*4)
    '''
    # apply filtering to odm_cls_target
    # arm_cls_preds_shape: (batch, 2 , num_anchors)
    arm_cls_preds = tf.nn.softmax(arm_cls_preds)
    arm_cls_preds_classes = tf.split(
        arm_cls_preds, num_or_size_splits=2, axis=1)
    arm_cls_preds_bg = tf.reshape(arm_cls_preds_classes[0], [0, -1])
    prob_threshold = tf.ones_like(arm_cls_preds_bg) * 0.99
    arm_hard_neg_cond = tf.greater_equal(arm_cls_preds_bg, prob_threshold)
    temp1 = tf.ones_like(odm_anchor_cls_label) * (-1)
    # filter out those well classified neg examples
    odm_cls_target_mask = tf.where(
        condition=arm_hard_neg_cond, x=temp1, y=odm_anchor_cls_label)

    # apply filtering to odm_loc_target_mask
    # odm_loc_target_mask_shape: (batch, num_anchors*4)

    arm_cls_preds_bg = tf.reshape(arm_cls_preds_bg, shape=[0, -1, 1])
    odm_loc_target_mask = tf.reshape(odm_loc_target_mask, shape=[0, -1, 4])
    odm_loc_target_mask = odm_loc_target_mask[:, :, 0]
    odm_loc_target_mask = tf.ones_like(
        odm_loc_target_mask, shape=[0, -1, 1])
    loc_temp = tf.ones_like(odm_loc_target_mask) * 0.99
    cond2 = tf.greater_equal(arm_cls_preds_bg, loc_temp)
    temp2 = tf.zeros_like(odm_loc_target_mask)
    odm_loc_target_bg_mask = tf.where(
        condition=cond2, x=temp2, y=odm_loc_target_mask)
    odm_loc_target_bg_mask = tf.concat(
        [odm_loc_target_bg_mask] * 4, axis=2)
    odm_loc_target_bg_mask = tf.reshape(
        odm_loc_target_bg_mask, shape=[0, -1])
    return [odm_cls_target_mask, odm_loc_target_bg_mask]
