import os
import time

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
            CreateDirs(self.GetMeshPath(modelName, 0))
            CreateDirs(self.GetSegmentationPath(modelName, 0))
            for modelName in self.models:
                for meshIdx in range(self.meshCnt[modelName]):
                    CreateDirs(self.GetDepthViewPath(modelName, meshIdx, 'test'))
                    CreateDirs(self.GetVertexViewPath(modelName, meshIdx, 'test'))
                    for segIdx in range(1):
                        CreateDirs(self.GetSegmentationViewPath(modelName, meshIdx, segIdx, 'test'))

    def AddModel(self, modelName, meshFormat, meshCnt):
        self.models.append(modelName)
        self.modelDir[modelName] = os.path.join(self.dataDir, modelName)
        self.meshFormat[modelName] = meshFormat
        self.meshCnt[modelName] = meshCnt

    def GetMeshPath(self, modelName, meshIdx):
        return os.path.join(self.modelDir[modelName], 'mesh', self.meshFormat[modelName].format(meshIdx))

    def GetSegmentationPath(self, modelName, segIdx, type='faceAs'):
        return os.path.join(self.modelDir[modelName], 'segmentation', '{:03d}-{}.npy'.format(segIdx, type))

    def GetVertexColorPath(self, modelName):
        return os.path.join(self.modelDir[modelName], 'vertexcolor.npy')

    def GetVisualColorPath(self, modelName):
        return os.path.join(self.modelDir[modelName], 'visualcolor.npy')

    def GetViewName(self, swi, dis, rot):
        return 's{:03d}_d{:03d}_r{:03d}'.format(swi, dis, rot)

    def GetDepthViewPath(self, modelName, meshIdx, viewName):
        return os.path.join(self.modelDir[modelName], 'view', 'mesh{:03d}'.format(meshIdx), 'depth', viewName + '.png')

    def GetVertexViewPath(self, modelName, meshIdx, viewName):
        return os.path.join(self.modelDir[modelName], 'view', 'mesh{:03d}'.format(meshIdx), 'vertex', viewName + '.exr')

    def GetSegmentationViewPath(self, modelName, meshIdx, segIdx, viewName):
        return os.path.join(self.modelDir[modelName], 'view', 'mesh{:03d}'.format(meshIdx), 'segmentation{:03d}'.format(segIdx), viewName + '.exr')


conf = Config('E:\\RealtimeHumanBodyCorrespondences')
conf.AddModel('SCAPE', 'mesh{:03d}.ply', 71)
conf.InitDirectories()