from config import *
import tensorflow as tf
import tensorflow.contrib.slim as slim


class RHBC:
    def __init__(self, mode, depth, labels=None, feat_opt=None, clas_opt=None):
        self.mode = mode
        self.depth = depth
        self.labels = labels
        self.feat_opt = feat_opt
        self.clas_opt = clas_opt

        self.build_feature(self.mode)

        if self.mode == 'test':
            return 

        self.build_classifier()

    def build_feature(self, mode):
        # alex fcn
        is_training = True if mode == 'train' else False
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], 
                            activation_fn=tf.nn.relu, 
                            padding='SAME'):#,
                            #normalizer_fn=slim.batch_norm, 
                            #normalizer_params={'is_training': is_training, 'decay': 0.95}):
            with tf.variable_scope('feature'):
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

                # with tf.variable_scope('decoder'):
                #     deconv8 = slim.conv2d_transpose(fc7, 256, [3, 3], 2)        # H/16
                #     deconv9 = slim.conv2d_transpose(deconv8, 256, [3, 3], 2)    # H/8
                #     deconv10 = slim.conv2d_transpose(deconv9, 96, [5, 5], 2)    # H/4
                #     deconv11 = slim.conv2d_transpose(deconv10, 16, [11, 11], 4)
                #     self.feature = deconv11

        self.feat_vars = slim.get_model_variables()

    def build_classifier(self):
        self.preds = {}
        self.clas_vars = {}
        self.losses = {}
        self.accuracy = {}
        self.train_ops = {}
        for model in conf.train_model_range:
            for seg in conf.train_seg_range:
                with tf.variable_scope('classifier-{}-{}'.format(model, seg)):
                    self.preds[(model, seg)] = slim.conv2d(self.feature, 500, [1, 1])
                    self.clas_vars[(model, seg)] = slim.get_model_variables()[-2:]

                with tf.variable_scope('losses-{}-{}'.format(model, seg)):
                    self.losses[(model, seg)], self.accuracy[(model, seg)] = self.loss_and_acc(self.labels, self.preds[(model, seg)])
                    grad = tf.gradients(self.losses[(model, seg)], self.feat_vars + self.clas_vars[(model, seg)])
                    train_op_feat = self.feat_opt.apply_gradients(zip(grad[:-2], self.feat_vars))
                    train_op_clas = self.clas_opt.apply_gradients(zip(grad[-2:], self.clas_vars[(model, seg)]))
                    self.train_ops[(model, seg)] = tf.group(train_op_feat, train_op_clas)

    def loss_and_acc(self, labels, logits):
        float_labels = tf.cast(labels, tf.float32)

        epsilon = tf.constant(value=1e-4)
        softmax = tf.nn.softmax(logits) + epsilon
        cross_entropy = -tf.reduce_sum(float_labels * tf.log(softmax), reduction_indices=[-1])
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        mask = tf.reduce_sum(float_labels, -1)
        predict = tf.argmax(softmax, -1)
        groundtruth = tf.argmax(labels, -1)
        correct_pixels = tf.reduce_sum(tf.multiply(mask, tf.cast(tf.equal(predict, groundtruth), tf.float32)))
        
        total_pixels = tf.constant(value=conf.width * conf.height, dtype=tf.float32)
        valid_pixels = tf.reduce_sum(float_labels)
        loss = tf.divide(tf.multiply(cross_entropy_mean, total_pixels), valid_pixels)
        acc = tf.divide(correct_pixels, valid_pixels)

        return loss, acc
                    