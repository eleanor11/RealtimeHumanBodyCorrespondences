from config import *
from meshutil import *
from colorutil import *
import gl.glm as glm
from gl.glrender import *
import time
import cv2 as cv

def GenSegmentation(modelName, segIdx, patches):
    sampleMeshPath = conf.MeshPath(modelName, 0)
    vertices, faces = LoadMesh(sampleMeshPath)
    centers, vertexAs, faceAs = FurthestPointSample(vertices, faces, patches, 10)

    datas = [centers, vertexAs, faceAs]
    types = ['center', 'vertexAs', 'faceAs']
    for i in range(3):
        segPath = conf.SegmentationPath(modelName, segIdx, types[i])
        np.save(segPath, datas[i])

def GenNewLabel(modelRange, segRange, swiRange, disRange, rotRange, depthAndVertex=True):
    b = ZFAR * ZNEAR / (ZNEAR - ZFAR)
    a = -b / ZNEAR

    # preparation
    if depthAndVertex:
        segRange = [-1] + segRange
    segColorMap = GetDistinctColors(NUMPATCH)

    renderer = GLRenderer(b'GenNewLabel', FULLSIZE, FULLSIZE, toTexture=True)
    proj = glm.perspective(glm.radians(70), 1.0, ZNEAR, ZFAR)
    for modelName in modelRange:
        vertexColor = None
        for meshIdx in range(conf.meshCnt[modelName]):
            print('Generate label for model {} Mesh {}...'.format(modelName, meshIdx))
            meshPath = conf.GetMeshPath(modelName, meshIdx)
            vertices, faces = LoadMesh(meshPath)
            RegularizeMesh(vertices, modelName)
            faces = faces.reshape([faces.shape[0] * 3])
            vertexBuffer = vertices[faces]

            if vertexColor is None:
                vertexColor = GetDistinctColors(vertices.shape[0])
            vertexColorBuffer = (vertexColor[faces] / 255.0).astype(np.float32)

            for segIdx in segRange + [-1]:
                # prepare segmentation color
                if segIdx != -1:
                    segPath = conf.GetSegmentationPath(modelName, segIdx)
                    segmentation = np.load(segPath)
                    segColorBuffer = np.zeros([faces.shape[0], 3], np.float32)
                    faceColors = segColorMap[segmentation] / 255.0
                    segColorBuffer[2::3,:] = segColorBuffer[1::3,:] = segColorBuffer[0::3,:] = faceColors

                for swi in swiRange:
                    for dis in disRange:
                        for rot in rotRange:
                            model = glm.identity()
                            model = glm.rotate(model, glm.radians(swi - MAXSWI / 2), glm.vec3(0, 1, 0))
                            model = glm.translate(model, glm.vec3(0, 0, -dis / 100.0))
                            model = glm.rotate(model, glm.radians(rot), glm.vec3(0, 1, 0))
                            mvp = proj.dot(model)

                            viewName = conf.GetViewName(swi, dis, rot)
                            if segIdx == -1:
                                rgb, z = renderer.draw(vertexBuffer, vertexColorBuffer, mvp.T)
                                # save depth view
                                depth = ((ZFAR - b / (z - a)) / (ZFAR - ZNEAR) * 255).astype(np.uint8)
                                dvPath = conf.GetDepthViewPath(modelName, meshIdx, viewName)
                                cv.imwrite(dvPath, depth)
                                # save vertex view
                                vertexIdx = ImageColor2Idx(rgb, vertices.shape[0] + 1)
                                vvPath = conf.GetVertexViewPath(modelName, meshIdx, viewName)
                                cv.imwrite(vvPath, vertexIdx)
                            else:
                                rgb, depth = renderer.draw(vertexBuffer, segColorBuffer, mvp.T)
                                # save segmentation view
                                seg = ImageColor2Idx(rgb, NUMPATCH + 1)
                                svPath = conf.GetSegmentationViewPath(modelName, meshIdx, segIdx, viewName)
                                cv.imwrite(svPath, seg)

if __name__ == '__main__':
    pass
    modelRange = ['SCAPE']
    segRange = [i for i in range(1)]
    swiRange = [35]
    disRange = [250]
    rotRange = [i for i in range(0, 360, 15)]
    GenNewLabel(modelRange, segRange, swiRange, disRange, rotRange, True)
