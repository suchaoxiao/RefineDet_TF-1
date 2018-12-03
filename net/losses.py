import tensorflow as tf


def abs_smooth(x):
    """Smoothed absolute function. Useful to compute an L1 smooth error.

    Define as:
        x^2 / 2         if abs(x) < 1
        abs(x) - 0.5    if abs(x) > 1
    We use here a differentiable definition using min(x) and abs(x). Clearly
    not optimal, but good enough for our purpose!
    """
    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)
    return r


def arm_losses(cls_preds_layers, loc_preds_layers,
           gclasses, glocalisations, gscores,
           match_threshold=0.5,
           negative_ratio=3.,
           alpha=1.,
           label_smoothing=0.,
           scope='arm_losses'):
    """Define the SSD network losses.
    """
    # since we only predict foreground/background, we convert all positive labels to 1
    gclasses = [tf.where(tf.greater(gclasses[i],0),tf.ones_like(gclasses[i]),gclasses[i])\
                 for i in range(len(cls_preds_layers))]
    return generate_losses(cls_preds_layers, loc_preds_layers,
                      gclasses, glocalisations, gscores,
                      match_threshold=match_threshold,
                      negative_ratio=negative_ratio,
                      alpha=alpha,
                      label_smoothing=label_smoothing,
                      scope=scope)

def odm_losses(cls_preds_layers, loc_preds_layers,
           gclasses, glocalisations, gscores,
           match_threshold=0.5,
           negative_ratio=3.,
           alpha=1.,
           label_smoothing=0.,
           scope='odm_losses'):
    """Define the SSD network losses.
    """
    return generate_losses(cls_preds_layers, loc_preds_layers,
                      gclasses, glocalisations, gscores,
                      match_threshold=match_threshold,
                      negative_ratio=negative_ratio,
                      alpha=alpha,
                      label_smoothing=label_smoothing,
                      scope=scope)

def generate_losses(cls_preds_layers, loc_preds_layers,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope=None):
    """Loss functions for training the SSD 300 VGG network.

    This function defines the different loss components of the SSD, and
    adds them to the TF loss collection.

    Arguments:
      logits: (list of) predictions logits Tensors;
      localisations: (list of) localisations Tensors;
      gclasses: (list of) groundtruth labels Tensors; - in batch
      glocalisations: (list of) groundtruth localisations Tensors;
      gscores: (list of) groundtruth score Tensors;
    """
    with tf.name_scope(scope, 'losses'):
        l_cross_pos = []
        l_cross_neg = []
        l_loc = []
        for ii, (loc_pred,cls_pred) in enumerate(zip(loc_preds_layers, cls_preds_layers)):
            dtype = cls_pred.dtype
            lshape = loc_pred.get_shape().as_list()
            cshape = cls_pred.get_shape().as_list()
            num_anchors = lshape[-1]//4
            num_classes = cshape[-1]//num_anchors
            loc_pred = tf.reshape(loc_pred,[-1,lshape[1]*lshape[2]*num_anchors,4])
            cls_pred = tf.reshape(cls_pred,[-1,cshape[1]*cshape[2]*num_anchors,num_classes])
            with tf.name_scope('block_%i' % ii):
                # Determine weights Tensor.
                pmask = gscores[ii] > match_threshold
                print('gscores['+str(ii)+']:', gscores[ii])
                print('gloc['+str(ii)+']:', glocalisations[ii])
                fpmask = tf.cast(pmask, dtype)
                n_positives = tf.reduce_sum(fpmask)

                # Select some random negative entries.
                # n_entries = np.prod(gclasses[i].get_shape().as_list())
                # r_positive = n_positives / n_entries
                # r_negative = negative_ratio * n_positives / (n_entries - n_positives)

                # Negative mask.
                no_classes = tf.cast(pmask, tf.int32)
                predictions = tf.nn.softmax(cls_pred)
                nmask = tf.logical_and(tf.logical_not(pmask),
                                       gscores[ii] > -0.5)
                fnmask = tf.cast(nmask, dtype)
                nvalues = tf.where(nmask,
                                   predictions[:, :, :, :, 0],
                                   1. - fnmask)
                nvalues_flat = tf.reshape(nvalues, [-1])
                # Number of negative entries to select.
                n_neg = tf.cast(negative_ratio * n_positives, tf.int32)
                n_neg = tf.maximum(n_neg, tf.size(nvalues_flat) // 8)
                n_neg = tf.maximum(n_neg, tf.shape(nvalues)[0] * 4)
                max_neg_entries = 1 + tf.cast(tf.reduce_sum(fnmask), tf.int32)
                n_neg = tf.minimum(n_neg, max_neg_entries)

                val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
                minval = val[-1]
                # Final negative mask.
                nmask = tf.logical_and(nmask, -nvalues > minval)
                fnmask = tf.cast(nmask, dtype)

                # Add cross-entropy loss.
                with tf.name_scope('cross_entropy_pos'):
                    # sparse loss accept label with 0-N rather than one-hot vectors
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_pred, 
                                                                          labels=gclasses[ii])
                    loss = tf.losses.compute_weighted_loss(loss, fpmask)
                    l_cross_pos.append(loss)

                with tf.name_scope('cross_entropy_neg'):
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_pred,
                                                                          labels=no_classes)
                    loss = tf.losses.compute_weighted_loss(loss, fnmask)
                    l_cross_neg.append(loss)

                # Add localization loss: smooth L1, L2, ...
                with tf.name_scope('localization'):
                    # Weights Tensor: positive mask + random negative.
                    weights = tf.expand_dims(alpha * fpmask, axis=-1)
                    loss = abs_smooth(
                        loc_pred - glocalisations[ii])
                    loss = tf.losses.compute_weighted_loss(loss, weights)
                    l_loc.append(loss)

        # Additional total losses...
        with tf.name_scope('total'):
            total_cross_pos = tf.add_n(l_cross_pos, 'cross_entropy_pos')
            total_cross_neg = tf.add_n(l_cross_neg, 'cross_entropy_neg')
            total_cross = tf.add(
                total_cross_pos, total_cross_neg, 'cross_entropy')
            total_loc = tf.add_n(l_loc, 'localization')
        return total_cross, total_loc
            # Add to EXTRA LOSSES TF.collection
            # tf.add_to_collection('EXTRA_LOSSES', total_cross_pos)
            # tf.add_to_collection('EXTRA_LOSSES', total_cross_neg)
            # tf.add_to_collection('EXTRA_LOSSES', total_cross)
            # tf.add_to_collection('EXTRA_LOSSES', total_loc)

