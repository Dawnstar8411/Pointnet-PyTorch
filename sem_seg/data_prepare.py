import os

import numpy as np
from path import Path

from utils.pc_util import save_h5
from .indoor3d_util import room2blocks_wrapper_normalized, collect_point_label

# -----------------------------------------------------------------------------
# 1. Collect indoor3d data.
# -----------------------------------------------------------------------------
data_path = Path('/home/yc/chen/data/Pointnet/S3DIS/Stanford3dDataset_v1.2_Aligned_Version')

with open('./meta/anno_paths.txt', "r") as f:
    anno_paths = [data_path / line.rstrip() for line in f.readlines()]


dump_path = Path('/home/yc/chen/data/Pointnet/S3DIS/stanford_indoor3d')
dump_path.makedirs_p()

# Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.

for anno_path in anno_paths:
    print(anno_path)
    try:
        elements = anno_path.split('/')
        out_filename = elements[-3] + '_' + elements[-2] + '.npy'  # Area_1_hallway_1.npy
        collect_point_label(anno_path, dump_path / out_filename, 'numpy')
    except:
        print(anno_path, 'ERROR!')

# -----------------------------------------------------------------------------
# 2. Generate indoor3d h5 format data.
# -----------------------------------------------------------------------------

indoor3d_data_dir = dump_path
num_point = 4096
h5_batch_size = 1000
data_dim = [num_point, 9]
label_dim = [num_point]
data_dtype = 'float32'
label_dtype = 'unit8'

with open('./meta/all_data_label.txt', "r") as f:
    data_label_files = [indoor3d_data_dir / line.rstrip() for line in f.readlines()]

dump_path = Path('/home/yc/chen/data/Pointnet/S3DIS/indoor3d_sem_seg_hdf5_data')
dump_path.makedirs_p()
output_filename_prefix = dump_path / 'ply_data_all'
output_room_filelist = dump_path / 'room_filelist.txt'

batch_data_dim = [h5_batch_size] + data_dim
batch_label_dim = [h5_batch_size] + label_dim
h5_batch_data = np.zeros(batch_data_dim, dtype=np.float32)
h5_batch_label = np.zeros(batch_label_dim, dtype=np.uint8)
buffer_size = 0  # state: record how many samples are currently in buffer
h5_index = 0  # state: the next h5 file to save


def insert_batch(data, label, last_batch=False):
    global h5_batch_data, h5_batch_label
    global buffer_size, h5_index
    data_size = data.shape[0]
    # If there is enough space, just insert
    if buffer_size + data_size <= h5_batch_data.shape[0]:
        h5_batch_data[buffer_size:buffer_size + data_size, ...] = data
        h5_batch_label[buffer_size:buffer_size + data_size] = label
        buffer_size += data_size
    else:  # not enough space
        capacity = h5_batch_data.shape[0] - buffer_size
        assert (capacity >= 0)
        if capacity > 0:
            h5_batch_data[buffer_size:buffer_size + capacity, ...] = data[0:capacity, ...]
            h5_batch_label[buffer_size:buffer_size + capacity, ...] = label[0:capacity, ...]
        # Save batch data and label to h5 file, reset buffer_size
        h5_filename = output_filename_prefix + '_' + str(h5_index) + '.h5'
        save_h5(h5_filename, h5_batch_data, h5_batch_label, data_dtype, label_dtype)
        print('Stored {0} with size {1}'.format(h5_filename, h5_batch_data.shape[0]))
        h5_index += 1
        buffer_size = 0
        # recursive call
        insert_batch(data[capacity:, ...], label[capacity:, ...], last_batch)
    if last_batch and buffer_size > 0:
        h5_filename = output_filename_prefix + '_' + str(h5_index) + '.h5'
        save_h5(h5_filename, h5_batch_data[0:buffer_size, ...], h5_batch_label[0:buffer_size, ...],
                data_dtype, label_dtype)
        print('Stored {0} with size {1}'.format(h5_filename, buffer_size))
        h5_index += 1
        buffer_size = 0
    return


sample_cnt = 0
f = open(output_room_filelist, 'w')
for i, data_label_filename in enumerate(data_label_files):
    print(data_label_filename)
    data, label = room2blocks_wrapper_normalized(data_label_filename, num_point, block_size=1.0,
                                                 stride=0.5,
                                                 random_sample=False, sample_num=None)
    print('{},{}'.format(data.shape, label.shape))
    for _ in range(data.shape[0]):
        f.write(os.path.basename(data_label_filename)[0:-4] + '\n')

    sample_cnt += data.shape[0]
    insert_batch(data, label, i == len(data_label_files) - 1)

f.close()
print("Total samples:{}".format(sample_cnt))
