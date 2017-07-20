from config import *
import os
from meshutil import *
from gl.glrender import *
import math
import numpy as np
from colorutil import *
import gl.glm as glm
import cv2 as cv

def RenameScapeMeshes():
    modelName = 'SCAPE'
    for i in range(52, 72):
        srcPath = conf.MeshPath(modelName, i)
        dstPath = conf.MeshPath(modelName, i - 1)
        os.rename(srcPath, dstPath)

if __name__ == '__main__':
    pass
