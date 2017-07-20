from config import *
import cv2 as cv
import numpy as np

def RandomBatch(batchSize, modelRange, segRange, swiRange, disRange, rotRange, mode='FULL'):
    models = np.random.choice(modelRange, batchSize)
    meshes = np.array([np.random.randint(0, conf.meshCnt[model]) for model in models])
    segs = np.random.choice(segRange, batchSize)
    swis = np.random.choice(swiRange, batchSize)
    diss = np.random.choice(disRange, batchSize)
    rots = np.random.choice(rotRange, batchSize)

    if mode == MODEFAST:
        batchDepth = np.zeros([batchSize, FASTSIZE, FASTSIZE, 1], np.uint8)
        batchLabel = np.zeros([batchSize, 1, 1, 500], np.int32)
        for i in range(batchSize):
            # load depth
            viewName = conf.ViewName(swis[i], diss[i], rots[i])
            dvPath = conf.DepthViewPath(models[i], meshes[i], viewName)
            HALFSIZE = FASTSIZE // 2
            depth = np.zeros([FULLSIZE + 2 * HALFSIZE, FULLSIZE + 2 * HALFSIZE], np.uint8)
            depth[HALFSIZE:HALFSIZE + FULLSIZE, HALFSIZE:HALFSIZE + FULLSIZE] = cv.imread(dvPath, -1)
            x, y = -1, -1
            while depth[x + HALFSIZE, y + HALFSIZE] == 0: x, y = np.random.randint(0, FULLSIZE), np.random.randint(0, FULLSIZE)
            print(x, y)
            batchDepth[i, :, :, 0] = depth[x: x + FASTSIZE, y: y + FASTSIZE]
            # load label
            svPath = conf.SegmentationViewPath(models[i], meshes[i], segs[i], viewName)
            segView = cv.imread(svPath, -1) + CVINTOFFSET
            batchLabel[i, 0, 0, segView[x, y] - 1] = 1
    else:
        batchDepth = np.zeros([batchSize, FULLSIZE, FULLSIZE, 1], np.uint8)
        batchLabel = np.zeros([batchSize, FULLSIZE, FULLSIZE, NUMPATCH], np.int32)
        for i in range(batchSize):
            # load depth
            viewName = conf.ViewName(swis[i], diss[i], rots[i])
            dvPath = conf.DepthViewPath(models[i], meshes[i], viewName)
            batchDepth[i, :, :, 0] = cv.imread(dvPath, -1)
            #load label
            svPath = conf.SegmentationViewPath(models[i], meshes[i], segs[i], viewName)
            segView = cv.imread(svPath, -1) + CVINTOFFSET
            mask = segView > 0
            batchLabel[i, mask, segView[mask] - 1] = 1

    return batchDepth, batchLabel

if __name__ == '__main__':
    modelRange = ['SCAPE']
    segRange = [0]
    swiRange = [35]
    disRange = [250]
    rotRange = [i for i in range(0, 360, 15)]
    mode = MODEFAST
