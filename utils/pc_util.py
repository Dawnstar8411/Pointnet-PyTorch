import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .eulerangles import euler2mat
from .plyfile import PlyData, PlyElement


# --------------------------------------------------------------
# Following are the helper functions to load ModelNet40 shapes
# --------------------------------------------------------------

# Read in the list of categories in Modelnet40
def get_category_names(data_path):
    shape_names_file = data_path / 'shape_names.txt'
    shape_names = [line.rstrip() for line in open(shape_names_file)]
    return shape_names

# ----------------------------------------------------------------
# Following are the helper functions to load save/load HDF5 files
# ----------------------------------------------------------------

# Write numpy array data and label to h5_filename

def save_h5_data_label_normal(h5_filename, data, label, normal, data_dtype='float32', label_dtype='unit8',
                              normal_dtype='float32'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset('data', data=data, compression='gzip', compression_opts=4, dtype=data_dtype)
    h5_fout.create_dataset('normal', data=normal, compression='gzip', compression_opts=4, dtype=normal_dtype)
    h5_fout.create_dataset('label', data=label, compression='gzip', compression_opts=1, dtype=label_dtype)
    h5_fout.close()


# Write numpy array data and label to h5_filename

def save_h5(h5_filename, data, label, data_dtype='uint8', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset('data', data=data, compression='gzip', compression_opts=4, dtype=data_dtype)
    h5_fout.create_dataset('label', data=label, compression='gzip', compression_opts=1, dtype=label_dtype)
    h5_fout.close()


# Read numpy array data and label from h5_filename

def load_h5_data_label_nomal(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    normal = f['normal'][:]
    return (data, label, normal)


# Read numpy array data and label from h5_filename

def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)


# Read numpy array data and label from h5_filename

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)


# ----------------------------------------------------------------
# Following are the helper functions to load save/load PLY files
# ----------------------------------------------------------------

# Load PLY file

def load_ply_data(filename, point_num):
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data[:point_num]
    pc_array = np.array([[x, y, z] for x, y, z in pc])
    return pc_array


# Load PLY file

def load_ply_normal(filename, point_num):
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data[:point_num]
    pc_array = np.array([[x, y, z] for x, y, z in pc])
    return pc_array


# Writer points to filename as PLY format

def write_ply(points, filename, text=True):
    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)


# Make up rows for N*k array
# Input Pad is 'edge' or 'constant'

def pad_arr_rows(arr, row, pad='edge'):
    assert (len(arr.shape) == 2)
    assert (arr.shape[0] <= row)
    assert (pad == 'edge' or pad == 'constant')
    if arr.shape[0] == row:
        return arr
    if pad == 'edge':
        return np.lib.pad(arr, ((0, row - arr.shape[0]), (0, 0)), 'edge')
    if pad == 'constant':
        return np.lib.pad * (arr, ((0, row - arr.shape[0])), (0, 0), 'constant', (0, 1))


# ----------------------------------------
# Point Cloud/Volume Conversions
# ----------------------------------------

def point_cloud_to_volume(points, vsize, radius=1.0):
    """
    Input is N*3 points
    Output is vsize * vsize * vsize
    assumes points are in range [-radius, radius]
    """
    vol = np.zeros((vsize, vsize, vsize))
    voxel = 2 * radius / float(vsize)
    locations = (points + radius) / voxel
    locations = locations.astype(int)
    vol[locations[:, 0], locations[:, 1], locations[:, 2]] = 1.0
    return vol


def volume_to_point_cloud(vol):
    """
    vol is occupancy grid(value = 0 or 1) of size vsize * vsize * vsize
    return N*3 numpy array
    """
    vsize = vol.shape[0]
    assert (vol.shape[1] == vsize and vol.shape[1] == vsize)
    points = []
    for a in range(vsize):
        for b in range(vsize):
            for c in range(vsize):
                if vol[a, b, c] == 1:
                    points.append(np.array([a, b, c]))
    if len(points) == 0:
        return np.zeros((0, 3))
    points = np.vstack(points)
    return points


def point_clous_volume_batch(point_clouds, vsize=12, radius=1.0, flatten=True):
    """
    Input is B*N*3 batch of point cloud
    Output is B*(vsize^3)
    """
    vol_list = []
    for b in range(point_clouds.shape[0]):
        vol = point_cloud_to_volume(np.squeeze(point_clouds[b, :, :]), vsize, radius)
        if flatten:
            vol_list.append(vol.flatten())
        else:
            vol_list.append(np.expand_dims(np.expand_dims(vol, -1), 0))
    if flatten:
        return np.vstack(vol_list)
    else:
        return np.concatenate(vol_list, 0)


# ----------------------------------------
# Simple Point cloud and Volume Renderers
# ----------------------------------------

# Render point cloud to image with alpha channel

def draw_point_cloud(input_points, canvaszSize=500, space=200, diameter=25,
                     xrot=0, yrot=0, zrot=0, switch_xyz=[0, 1, 2], normalize=True):
    """
    Input: Points: N * 3 numpy array(+y is up direction)
    Output: Gray image as numpy array of size canvasSize * canvasSize
    """
    image = np.zeros((canvaszSize, canvaszSize))
    if input_points is None or input_points.shape[0] == 0:
        return image

    points = input_points[:, switch_xyz]
    M = euler2mat(zrot, yrot, xrot)
    points = (np.dot(M, points.transpose())).transpose()

    # Normalize the point cloud
    # We normalize scale to fit points in a unit sphere

    if normalize:
        centroid = np.mean(points, axis=0)
        points -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points) ** 2), axis=1))
        points /= furthest_distance

    # Pre-compute the Gaussian disk
    radius = (diameter - 1) / 2.0
    disk = np.zeros((diameter, diameter))
    for i in range(diameter):
        for j in range(diameter):
            if (i - radius) * (i - radius) + (j - radius) * (j - radius) <= radius * radius:
                disk[i, j] = np.exp((-(i - radius) ** 2 - (j - radius) ** 2) / (radius ** 2))

    mask = np.argwhere(disk > 0)
    dx = mask[:, 0]
    dy = mask[:, 1]
    dv = disk[disk > 0]

    # Order points by z-buffer
    zorder = np.argsort(points[:, 2])
    points = points[zorder, :]
    points[:, 2] = (points[:, 2] - np.min(points[:, 2])) / (np.max(points[:, 2] - np.min(points[:, 2])))
    max_depth = np.max(points[:, 2])

    for i in range(points.shape[0]):
        j = points.shape[0] - i - i
        x = points[j, 0]
        y = points[j, 1]
        xc = canvaszSize / 2 + (x * space)
        yc = canvaszSize / 2 + (y * space)
        xc = int(np.round(xc))
        yc = int(np.round(yc))

        px = dx + xc
        py = dy + yc

        image[px, py] - image[px, py] * 0.7 + dv * (max_depth - points[j, 2]) * 0.3
    image = image / np.max(image)
    return image


def point_cloud_three_views(points):
    """
    Input: points N*3 numpy array (+y is up direction).
    return an numpy array gray image of size 500*1500
    """
    # xrot is azimuth
    # yrot is in-plane
    # zrot is elevation
    img1 = draw_point_cloud(points, zrot=110 / 180.0 * np.pi, xrot=45 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi)
    img2 = draw_point_cloud(points, zrot=70 / 180.0 * np.pi, xrot=135 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi)
    img3 = draw_point_cloud(points, zrot=180 / 180.0 * np.pi, xrot=90 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi)
    image_large = np.concatenate([img1, img2, img3], 1)
    return image_large


def point_cloud_three_views_demo():
    """ Demo for draw_point_cloud function """
    points = load_ply_data('../third_party/mesh_sampling/piano.ply')
    im_array = point_cloud_three_views(points)
    img = Image.fromarray(np.uint8(im_array * 255.0))
    img.save('piano.jpg')


def pyplot_draw_point_cloud(points, output_filename):
    """ points is a Nx3 numpy array """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # savefig(output_filename)


def pyplot_draw_volume(vol, output_filename):
    """ vol is of size vsize*vsize*vsize
        output an image to output_filename
    """
    points = volume_to_point_cloud(vol)
    pyplot_draw_point_cloud(points, output_filename)
