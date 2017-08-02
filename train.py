from config import *
import cv2 as cv
import numpy as np
from net import *
from colorutil import *

def decode_label(seg_view):
    label = np.zeros([conf.height, conf.width, conf.num_classes], np.uint8)
    seg = image_color2idx(seg_view, gbr=True)
    mask = seg > 0
    label[mask, seg[mask] - 1] = 1
    return label

def random_batch(model_range, seg_range, swi_range, dis_range, rot_range, batch_size):
    # create_tmp_list
    file_list = []
    for model in model_range:
        for mesh in range(conf.num_meshes[model]):
            for seg in seg_range:
                for swi in swi_range:
                    for dis in dis_range:
                        for rot in rot_range:
                            view_name = conf.view_name(swi, dis, rot)
                            depth_view_path = conf.depth_view_path(model, mesh, view_name)
                            seg_view_path = conf.segmentation_view_path(model, mesh, seg, view_name)
                            file_list.append('{} {}\n'.format(depth_view_path, seg_view_path))

    np.random.shuffle(file_list)
    tmp_file_path = 'file_list.txt'
    with open(tmp_file_path, 'w') as file:
        for line in file_list:
            file.write(line)

    input_queue = tf.train.string_input_producer([tmp_file_path])
    line_reader = tf.TextLineReader()
    _, line = line_reader.read(input_queue)
    data_paths = tf.string_split([line]).values

    depth = tf.image.decode_png(tf.read_file(data_paths[0]), 0)

    seg = tf.image.decode_png(tf.read_file(data_paths[1]), 0)
    label = tf.py_func(decode_label, [seg], [tf.uint8])[0]

    depth.set_shape([conf.height, conf.width, 1])
    label.set_shape([conf.height, conf.width, conf.num_classes])

    depth = tf.cast(depth, tf.float32)

    depth_batch, label_batch = tf.train.batch([depth, label], batch_size, num_threads=8)
    return depth_batch, label_batch

def train(model_range, seg_range, swi_range, dis_range, rot_range, batch_size=8, num_epochs=10, learning_rate=1e-4, log_dir='', checkpoint_path='', num_gpus=1):
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.Variable(0, trainable=False)

        num_training_samples = len(seg_range) * len(swi_range) * len(dis_range) * len(rot_range) * np.sum([conf.num_meshes[model] for model in model_range])
        print(num_training_samples)
        steps_per_epoch = np.ceil(num_training_samples / batch_size).astype(np.int32)

        num_total_steps = num_epochs * steps_per_epoch
        start_learning_rate = learning_rate

        boundaries = [np.int32((3/5) * num_total_steps), np.int32((4/5) * num_total_steps)]
        values = [learning_rate, learning_rate / 2, learning_rate / 4]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        
        opt_step = tf.train.AdamOptimizer(learning_rate)

        depth_batch, label_batch = random_batch(model_range, seg_range, swi_range, dis_range, rot_range, batch_size)

        depth_batch_splits = tf.split(depth_batch, num_gpus, 0)
        label_batch_splits = tf.split(label_batch, num_gpus, 0)

        tower_grads = []
        tower_losses = []
        reuse_variables = None
       
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(num_gpus):
                with tf.device('/gpu:%d' % i):

                    net = RHBC('train', depth_batch_splits[i], label_batch_splits[i], reuse_variables, i)

                    # loss = model.total_loss
                    # tower_losses.append(loss)

                    # reuse_variables = True

                    # grads = opt_step.compute_gradients(loss)

                    # tower_grads.append(grads)

        # grads = average_gradients(tower_grads)

        # apply_gradient_op = opt_step.apply_gradients(grads, global_step=global_step)

        # total_loss = tf.reduce_mean(tower_losses)
        
        # tf.summary.scalar('learning_rate', learning_rate, ['model_0'])
        # tf.summary.scalar('total_loss', total_loss, ['model_0'])

        # with tf.Session() as sess:
        #     coordinator = tf.train.Coordinator()
        #     threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        #     coordinator.request_stop()
        #     coordinator.join(threads)

if __name__ == '__main__':
    np.random.seed()
    model_range = ['SCAPE']
    seg_range = [0]
    swi_range = [35]
    dis_range = [250]
    rot_range = [i for i in range(0, 360, 15)]
    batch_size = 1
    train(model_range, seg_range, swi_range, dis_range, rot_range,
        batch_size=batch_size,
        num_epochs=10,
        learning_rate=1e-4,
        log_dir='',
        checkpoint_path='',
        num_gpus=1
        )
