from config import *
import numpy as np

def idx2color(idx):
    r = idx // (256 * 256) % 256
    g = idx // 256 % 256
    b = idx % 256
    return np.array([r, g, b], dtype=np.uint8)

def image_color2idx(color_img, gbr=False):
    color_img = color_img.astype(np.int32)
    idx = np.zeros([color_img.shape[0], color_img.shape[1]], np.int32)
    if gbr:
        idx[:,:] += color_img[:,:,2] * 256 * 256
        idx[:,:] += color_img[:,:,1] * 256
        idx[:,:] += color_img[:,:,0]
    else:
        idx[:,:] += color_img[:,:,0] * 256 * 256
        idx[:,:] += color_img[:,:,1] * 256
        idx[:,:] += color_img[:,:,2]
    return idx

def distinct_colors(num_classes):
    colors = np.zeros([num_classes, 3], np.uint8)
    for i in range(num_classes):
        colors[i] = idx2color(i + 1)
    return colors