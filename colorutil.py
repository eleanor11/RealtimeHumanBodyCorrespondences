from config import *
import numpy as np

rgbRange = 256 * 256 * 256
def Idx2Color(idx, maxIdx):
    k = idx * (rgbRange // maxIdx)
    r = k // (256 * 256) % 256
    g = k // 256 % 256
    b = k % 256
    return np.array([r, g, b], dtype=np.uint8)

def Color2Idx(color, maxIdx):
    k = (color[0] * 256  * 256 + color[1] * 256 + color[2]) 
    idx = k // (rgbRange // maxIdx)
    return idx

def ImageColor2Idx(rgb, maxIdx):
    rgb = rgb.astype(np.int32)
    idx = np.zeros([rgb.shape[0], rgb.shape[1]], np.int32)
    idx[:,:] += rgb[:,:,0] * 256 * 256
    idx[:,:] += rgb[:,:,1] * 256
    idx[:,:] += rgb[:,:,2]
    idx //= (rgbRange // maxIdx)
    return idx

def DistinctColors(numClass):
    colors = np.zeros([numClass, 3], np.uint8)
    for i in range(numClass):
        colors[i] = Idx2Color(i + 1, numClass + 1)
    return colors