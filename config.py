import os
import time

MODEFAST = 'FAST'
MODEFULL = 'FULL'

ZNEAR = 1.0
ZFAR = 3.5
MAXSWI = 70
NUMPATCH = 500
FULLSIZE = 512
FASTSIZE = 225

CVINTOFFSET = 2147483648

def CreateDirs(path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def Timeit(t=None):
    if t:
        print(time.time() - t)
    return time.time()

class Config:
    def __init__(self, projectDir):
        self.projectDir = projectDir
        self.dataDir = os.path.join(self.projectDir, 'data')
        self.models = []
        self.modelDir = {}
        self.meshFormat = {}
        self.meshCnt = {}

    def InitDirectories(self):
        for modelName in self.models:
            CreateDirs(self.MeshPath(modelName, 0))
            CreateDirs(self.SegmentationPath(modelName, 0))
            for modelName in self.models:
                for meshIdx in range(self.meshCnt[modelName]):
                    CreateDirs(self.DepthViewPath(modelName, meshIdx, 'test'))
                    CreateDirs(self.VertexViewPath(modelName, meshIdx, 'test'))
                    for segIdx in range(1):
                        CreateDirs(self.SegmentationViewPath(modelName, meshIdx, segIdx, 'test'))

    def AddModel(self, modelName, meshFormat, meshCnt):
        self.models.append(modelName)
        self.modelDir[modelName] = os.path.join(self.dataDir, modelName)
        self.meshFormat[modelName] = meshFormat
        self.meshCnt[modelName] = meshCnt

    def MeshPath(self, modelName, meshIdx):
        return os.path.join(self.modelDir[modelName], 'mesh', self.meshFormat[modelName].format(meshIdx))

    def SegmentationPath(self, modelName, segIdx, type='faceAs'):
        return os.path.join(self.modelDir[modelName], 'segmentation', '{:03d}-{}.npy'.format(segIdx, type))

    def VisualColorPath(self, modelName):
        return os.path.join(self.modelDir[modelName], 'visualcolor.npy')

    def ViewName(self, swi, dis, rot):
        return 's{:03d}_d{:03d}_r{:03d}'.format(swi, dis, rot)

    def DepthViewPath(self, modelName, meshIdx, viewName):
        return os.path.join(self.modelDir[modelName], 'view', 'mesh{:03d}'.format(meshIdx), 'depth', viewName + '.png')

    def VertexViewPath(self, modelName, meshIdx, viewName):
        return os.path.join(self.modelDir[modelName], 'view', 'mesh{:03d}'.format(meshIdx), 'vertex', viewName + '.exr')

    def SegmentationViewPath(self, modelName, meshIdx, segIdx, viewName):
        return os.path.join(self.modelDir[modelName], 'view', 'mesh{:03d}'.format(meshIdx), 'segmentation{:03d}'.format(segIdx), viewName + '.exr')


conf = Config('E:\\RealtimeHumanBodyCorrespondences')
conf.AddModel('SCAPE', 'mesh{:03d}.ply', 71)
conf.InitDirectories()