from config import *
import tensorflow as tf
import tensorflow.contrib.slim as slim


class RHBC:
    def __init__(self, mode, depth, label, reuse_variables=None, model_idx=0):
        self.mode = mode
        self.depth = depth
        self.label = label
        self.model_collection = ['model_' + str(model_idx)]

        self.reuse_variables = reuse_variables

        self.build_model()

        if self.mode == 'test':
            return 

        self.build_losses()
        self.build_summaries()

    def build_model(self):
        # alex fcn
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.relu, padding='SAME'):
            with tf.variable_scope('model', reuse=self.reuse_variables):
                with tf.variable_scope('encoder'):
                    conv1 = slim.conv2d(self.depth, 96, [11, 11], 4)            # H/4
                    pool1 = slim.max_pool2d(conv1, [3, 3], 2, padding='SAME')   # H/8
                    conv2 = slim.conv2d(pool1, 256, [5, 5], 1)                  # H/8
                    pool2 = slim.max_pool2d(conv2, [3, 3], 2, padding='SAME')   # H/16
                    conv3 = slim.conv2d(pool2, 384, [3, 3], 1)                  # H/16
                    conv4 = slim.conv2d(conv3, 384, [3, 3], 1)                  # H/16
                    conv5 = slim.conv2d(conv4, 256, [3, 3], 1)                  # H/16
                    pool5 = slim.max_pool2d(conv5, [3, 3], 2, padding='SAME')   # H/32
                    fc6 = slim.conv2d(pool5, 4096, [1, 1], 1)                   # H/32
                    fc7 = slim.conv2d(fc6, 4096, [1, 1], 1)                     # H/32

                with tf.variable_scope('skips'):
                    skip1 = conv1
                    skip2 = conv2
                    skip3 = conv5

                with tf.variable_scope('decoder'):
                    deconv8 = slim.conv2d_transpose(fc7, 256, [3, 3], 2)        # H/16
                    concat8 = tf.concat([deconv8, skip3], 3)

                    deconv9 = slim.conv2d_transpose(concat8, 256, [3, 3], 2)    # H/8
                    concat9 = tf.concat([deconv9, skip2], 3)

                    deconv10 = slim.conv2d_transpose(concat9, 96, [5, 5], 2)    # H/4
                    concat10 = tf.concat([deconv10, skip1], 3)

                    deconv11 = slim.conv2d_transpose(concat10, 16, [11, 11], 4)
                    self.feature = deconv11

    def build_losses(self):
        self.preds = {}
        self.losses = {}
        with tf.variable_scope('classifier', reuse=self.reuse_variables):
            for model in conf.train_model_range:
                for seg in conf.train_seg_range:
                    self.preds[(model, seg)] = slim.conv2d(self.feature, 500, [1, 1])
                    self.losses[(model, seg)] = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.preds[(model, seg)])