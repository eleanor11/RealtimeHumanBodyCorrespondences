from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import cv2 as cv
import numpy as np
from net import *
from colorutil import *
from meshutil import *

def sqrdis(src, dst):
    dists = np.multiply(np.dot(dst, src.T), -2)
    dists += np.sum(np.square(dst), 1, keepdims=True)
    dists += np.sum(np.square(src), 1)
    return dists

def load_model_color(model_color_path):
    with open(model_color_path, 'r') as file:
        colors = file.read().split('\n')[:-1]
        colors = list(map(lambda s: list(reversed(list(map(lambda x: int(255 * float(x)), s.split(' '))))), colors))
    return np.array(colors)

class Pmohb(object):
    def __init__(self, net_ckpt_path, model, model_feat_path):
        print("Start initializing Pmohb...")
        self.znear = 1.0
        self.zfar = 3.5

        # init kinect
        self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)

        # init network
        with tf.Graph().as_default():
            self.depth_input = tf.placeholder(tf.float32, [1, 512, 512, 1])
            self.net = RHBC('test', self.depth_input, None, None, None)

            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            saver = tf.train.Saver(self.net.feat_vars)
            saver = tf.train.Saver()
            saver.restore(self.sess, net_ckpt_path)

        # init model
        print("Prepare model...")

        model_color_path = conf.model_color_path(model)
        self.model_color = load_model_color(model_color_path)

        self.model_feat = np.load(model_feat_path)
        sample_mesh_path = conf.mesh_path(model, 0)
        vertices, faces = load_mesh(sample_mesh_path)
        num_vertex = vertices.shape[0]
        
        num_clas = 64
        centroids, belong, _ = furthest_point_sample(vertices, faces, num_clas, 5)
        self.key_feat = self.model_feat[centroids]
        self.clas_vert = []
        self.clas_feat = []
        for i in range(num_clas):
            clas_mask = belong == i
            clas_v = np.arange(num_vertex)[clas_mask]
            self.clas_vert.append(clas_v)
            self.clas_feat.append(self.model_feat[clas_v])

        # init camera and projector params
        self.cx = 263.088
        self.cy = 207.137
        self.fx = 365.753
        self.fy = 365.753
        self.t = [
            -1.390668693623873,
-0.011007482582276377,
-0.8667484528337432,
0.6619667407488129,
-0.03976989021546296,
1.7952502894980042,
-1.2026607080803946,
0.6863956127031143,
-0.0337235987568367,
0.004993447985183048,
-1.5366058015071355
            ]
        self.pwidth = 1024
        self.pheight = 768

        self.background = np.zeros([424 * 512], np.float32)

    def record_background(self, num_frame):
        print("Record {} frames as background...".format(num_frame))
        cnt = np.zeros([424 * 512], np.uint8)
        for i in range(num_frame):
            while 1:
                if self.kinect.has_new_depth_frame():
                    frame = self.kinect.get_last_depth_frame()
                    cnt[frame > 0] += 1
                    self.background += frame
                    break
        mask = cnt > 0
        self.background[mask] /= cnt[mask]

    # raw depth, (512*424,), depth in millimeter
    def preprocess(self, rawdepth):
        rawdepth[self.background - rawdepth < 50] = self.zfar * 1000
        rawdepth[rawdepth == 0] = self.zfar * 1000
        rawdepth = rawdepth / 1000

        rawdepth[rawdepth > self.zfar] = self.zfar
        rawdepth[rawdepth < self.znear] = self.zfar

        depth = ((self.zfar - rawdepth) / (self.zfar - self.znear) * 255).astype(np.uint8)
        depth = depth.reshape([424, 512, 1])
        return depth

    def match_all(self, depth, mask):
        depth_cmp = np.zeros([1, 512, 512, 1], np.uint8)
        depth_cmp[0, 44:468, :, 0] = depth.reshape([424, 512])
        img_feat = self.sess.run(self.net.feature, feed_dict={self.depth_input: depth_cmp}).reshape([512, 512, 16])[44:468, :, :]

        feat = img_feat[mask]
        dists = sqrdis(self.model_feat, feat)
        match = np.argmin(dists, -1)

        corres = np.zeros([424, 512, 3], np.uint8)
        corres[mask] = self.model_color[match]
        return corres


    def match(self, depth, mask):
        depth_cmp = np.zeros([1, 512, 512, 1], np.uint8)
        depth_cmp[0, 44:468, :, 0] = depth.reshape([424, 512])
        img_feat = self.sess.run(self.net.feature, feed_dict={self.depth_input: depth_cmp}).reshape([512, 512, 16])[44:468, :, :]

        feat = img_feat[mask]
        key_dists = sqrdis(self.key_feat, feat)
        N = 1
        nearest = np.argpartition(key_dists, N, -1)[:, :N]

        all_match = np.zeros([feat.shape[0], N], np.int64)
        all_dists = np.ones([feat.shape[0], N]) * 1e10
        for i in range(self.key_feat.shape[0]):
            for j in range(N):
                clas_mask = nearest[:, j] == i
                if clas_mask.shape[0] > 0:
                    dst_feat = feat[clas_mask]
                    dists = sqrdis(self.clas_feat[i], dst_feat)
                    min_dist = np.amin(dists, -1)
                    match = np.argmin(dists, -1)
                    all_dists[clas_mask, j] = min_dist
                    all_match[clas_mask, j] = self.clas_vert[i][match]
        min_dists = np.argmin(all_dists, -1)
        match = all_match[np.arange(all_match.shape[0]), min_dists]

        corres = np.zeros([424, 512, 3], np.uint8)
        corres[mask] = self.model_color[match]

        return corres

    def project(self, rawdepth, corres, mask):
        rawdepth = rawdepth.reshape([424, 512])[mask]
        u = np.array([[i for i in range(511, -1, -1)]] * 424)[mask]
        v = np.array([[i for j in range(512)] for i in range(424)])[mask]
        color = corres[mask]

        Z = rawdepth / 1000
        X = (u - self.cx) * Z / self.fx
        Y = (self.cy - v) * Z / self.fy

        t = self.t
        denom = t[8] * X + t[9] * Y + t[10] * Z + 1
        #x = (t[0] * X + t[1] * Y + t[2] * Z + t[3]) / denom * 2 - 1
        #y = 1 - (t[4] * X + t[5] * Y + t[6] * Z + t[7]) / denom * 2
        x = self.pwidth * (t[0] * X + t[1] * Y + t[2] * Z + t[3]) / denom
        y = self.pheight * (t[4] * X + t[5] * Y + t[6] * Z + t[7]) / denom

        x = x.astype(np.int32)
        x[x < 0] = 0
        x[x >= self.pwidth] = 0
        y = y.astype(np.int32)
        y[y < 0] = 0
        y[y >= self.pheight] = 0

        proj = np.zeros([self.pheight, self.pwidth, 3], np.uint8)
        proj[y, x] = color
        cv.imshow('projection', proj)
        cv.moveWindow('projection', 1920, 0)


    def run(self):
        while 1:
            ch = cv.waitKey(1)
            if ch == 27:
                break

            if self.kinect.has_new_depth_frame():
                # frame, raw depth, [424 * 512 * 1], depth in mm
                rawdepth = self.kinect.get_last_depth_frame()

                # preprocess, [512, 512, 1], gray image
                depth = self.preprocess(rawdepth)
                mask = depth.reshape([424, 512]) > 0
                cv.imshow('depth', depth)

                # extract feature
                corres = self.match(depth, mask)
                #corres = self.match_all(depth, mask)
                cv.imshow('correspond', corres)

                self.project(rawdepth, corres, mask)



                # proj = 255 * np.ones([768, 1024, 1], np.uint8)
                # cv.imshow('proj', proj)
                # cv.moveWindow('proj', 1920, 0)
                # cv.setWindowProperty("proj", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN);

if __name__ == '__main__':
    core = Pmohb('log/alex-skip-1-5-2/model-85200', 'SCAPE', 'data/SCAPE/feature/alex-skip-1-5/model-85200.npy')
    core.record_background(60)
    core.run()