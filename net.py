import tensorflow as tf
import tensorflow.contrib.slim as slim

class DHBC:
    def Inference(self, mode):
        if mode == 'FAST':
            padding = 'VALID'
            self.depth = tf.placeholder(tf.float32, [None, 226, 226, 1])
            self.conv1 = slim.conv2d(depth, 96, [11, 11], 4, padding, scope='conv1')
            self.pool1 = slim.max_pool2d(self.conv1, [3, 3], 2, scope='pool1')
            self.conv2 = slim.conv2d(self.pool1, 256, [5, 5], 1, padding, scope='conv2')
            self.pool2 = slim.max_pool2d(self.conv2, [3, 3], 2, scope='pool2')
            self.conv3 = slim.conv2d(self.pool2, 384, [3, 3], 1, padding, scope='conv3')
            self.conv4 = slim.conv2d(self.conv3, 384, [3, 3], 1, padding, scope='conv4')
            self.conv5 = slim.conv2d(self.conv4, 256, [3, 3], 1, padding, scope='conv5')
            self.pool5 = slim.max_pool2d(self.conv5, [3, 3], 2, scope='pool5')
            self.fc6 = slim.conv2d(self.pool5, 4096, [1, 1], 1, padding, scope='fc6')
            self.fc7 = slim.conv2d(self.fc6, 4096, [1, 1], 1, padding, scope='fc7')
            self.conv8 = slim.conv2d_transpose(self.fc7, 16, [3, 3], 1, 'SAME', scope='conv8')
            self.feature = self.conv8
        else:
            padding = 'SAME'
            self.depth = tf.placeholder(tf.float32, [None, 512, 512, 1])
            self.conv1 = slim.conv2d(depth, 96, [11, 11], 4, padding, scope='conv1')
            self.conv2 = slim.conv2d(self.conv1, 256, [5, 5], 1, padding, scope='conv2')
            self.conv3 = slim.conv2d(self.conv2, 384, [3, 3], 1, padding, scope='conv3')
            self.conv4 = slim.conv2d(self.conv3, 384, [3, 3], 1, padding, scope='conv4')
            self.conv5 = slim.conv2d(self.conv4, 256, [3, 3], 1, padding, scope='conv5')
            self.fc6 = slim.conv2d(self.conv5, 4096, [1, 1], 1, padding, scope='fc6')
            self.fc7 = slim.conv2d(self.fc6, 4096, [1, 1], 1, padding,scope='fc7')
            self.conv8 = slim.conv2d_transpose(self.fc7, 16, [3, 3], 4, padding, scope='conv8')
            self.feature = self.conv8

        return self.depth, self.feature