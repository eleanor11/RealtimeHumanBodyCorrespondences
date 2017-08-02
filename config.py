import os
import time

def create_dirs(path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def Timeit(t=None):
    if t:
        print(time.time() - t)
    return time.time()

class Config:
    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.data_dir = os.path.join(self.project_dir, 'data')
        self.models = []
        self.model_dir = {}
        self.mesh_format = {}
        self.num_meshes = {}

        # constant
        self.cv_int_offset = 2147483648
        self.width = 512
        self.height = 512
        self.num_classes = 500
        self.znear = 1.0
        self.zfar = 3.5
        self.max_swi = 70

    def init_directories(self):
        for model in self.models:
            create_dirs(self.mesh_path(model, 0))
            create_dirs(self.segmentation_path(model, 0))
            for model in self.models:
                for mesh_idx in range(self.num_meshes[model]):
                    create_dirs(self.depth_view_path(model, mesh_idx, 'test'))
                    create_dirs(self.vertex_view_path(model, mesh_idx, 'test'))
                    for seg_idx in range(1):
                        create_dirs(self.segmentation_view_path(model, mesh_idx, seg_idx, 'test'))

    def add_model(self, model, mesh_format, num_meshes):
        self.models.append(model)
        self.model_dir[model] = os.path.join(self.data_dir, model)
        self.mesh_format[model] = mesh_format
        self.num_meshes[model] = num_meshes

    def mesh_path(self, model, mesh_idx):
        return os.path.join(self.model_dir[model], 'mesh', self.mesh_format[model].format(mesh_idx))

    def segmentation_path(self, model, seg_idx, type='face_as'):
        return os.path.join(self.model_dir[model], 'segmentation', '{:03d}_{}.npy'.format(seg_idx, type))

    def view_name(self, swi, dis, rot):
        return 's{:03d}_d{:03d}_r{:03d}'.format(swi, dis, rot)

    def depth_view_path(self, model, mesh_idx, view_name):
        return os.path.join(self.model_dir[model], 'view', 'mesh{:03d}'.format(mesh_idx), 'depth', view_name + '.png')

    def vertex_view_path(self, model, mesh_idx, view_name):
        return os.path.join(self.model_dir[model], 'view', 'mesh{:03d}'.format(mesh_idx), 'vertex', view_name + '.png')

    def segmentation_view_path(self, model, mesh_idx, seg_idx, view_name):
        return os.path.join(self.model_dir[model], 'view', 'mesh{:03d}'.format(mesh_idx), 'segmentation{:03d}'.format(seg_idx), view_name + '.png')


conf = Config('E:\\RealtimeHumanBodyCorrespondences')
conf.add_model('SCAPE', 'mesh{:03d}.ply', 71)
conf.init_directories()