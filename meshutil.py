from config import *
from plyfile import PlyData, PlyElement
import numpy as np
import gl.glm as glm

def SqrDist(src, dst):
    sqrDists = np.multiply(np.dot(dst, src.T), -2)
    sqrDists += np.sum(np.square(dst), 1, keepdims=True)
    sqrDists += np.sum(np.square(src), 1)
    return sqrDists

def LoadPlyMesh(meshPath):
    data = PlyData.read(meshPath)

    vertexData = data['vertex'].data
    vertices = np.zeros([vertexData.shape[0], 3], dtype=np.float32)
    for i in range(vertices.shape[0]):
        for j in range(3):
            vertices[i, j] = vertexData[i][j]

    faceData = data['face'].data
    faces = np.zeros([faceData.shape[0], 3], dtype=np.int32)
    for i in range(faces.shape[0]):
        for j in range(3):
            faces[i, j] = faceData[i][0][j]

    return vertices, faces

def LoadMesh(meshPath):
    if meshPath.endswith('.ply'):
        vertices, faces = LoadPlyMesh(meshPath)
    if np.min(faces) == 1:
        faces -= 1
    return vertices, faces

def RegularizeMesh(vertices, modelName):
    tmp = np.ones([vertices.shape[0], 4], dtype=np.float32)
    tmp[:,:3] = vertices

    if modelName.startswith('SCAPE'):
        m = glm.identity()
        m = glm.rotate(m, glm.radians(90), glm.vec3(0, 0, 1))
        m = glm.rotate(m, glm.radians(270), glm.vec3(1, 0, 0))
        tmp = glm.transform(tmp, m)
    elif modelName.startswith('MIT'):
        pass

    vertices[:,:] = tmp[:,:3]

    mean = np.mean(vertices, 0)
    vertices -= mean

def FurthestPointSample(vertices, faces, N, K):
    numVertex = vertices.shape[0]
    centerIndices = np.random.choice(numVertex, N, replace=False)
    sqrDists = 1e10 * np.ones(numVertex)
    vertexAs = np.zeros(numVertex, dtype=np.int32)
    for i in range(N):
        newSqrDists = np.sum(np.square(vertices - vertices[centerIndices[i]]), 1)
        updateMask = newSqrDists < sqrDists
        sqrDists[updateMask] = newSqrDists[updateMask]
        vertexAs[updateMask] = i
        nextCenter = np.argmax(sqrDists)
        if K - 1 <= i < N - 1:
            centerIndices[i + 1] = nextCenter

    centers = vertices[centerIndices]
    faceCenters = np.mean(vertices[faces], 1)
    sqrDists = SqrDist(centers, faceCenters)
    faceAs = np.argmin(sqrDists, 1)
    return centerIndices, vertexAs, faceAs
