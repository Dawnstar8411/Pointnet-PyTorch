import random
import torch
import numpy as np


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, points):
        for t in self.transforms:
            points = t(points)
        return points


class rotate_point_cloud(object):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, rotated batch of point clouds
    """

    def __call__(self, points):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        rotated_data = np.dot(points.reshape((-1, 3)), rotation_matrix)
        return rotated_data


class rotate_point_cloud_by_angle(object):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, rotated batch of point clouds
    """

    def __inti__(self, rotation_angle):
        self.rotation_angle = rotation_angle

    def __call__(self, points):
        cosval = np.cos(self.rotation_angle)
        sinval = np.sin(self.rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        rotated_data = np.dot(points.reshape((-1, 3)), rotation_matrix)
        return rotated_data


class jitter_point_cloud(object):
    """ Randomly jitter points. jittering is per point.
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, jittered batch of point clouds
    """

    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, points):
        N, C = points.shape
        assert (self.clip > 0)
        jittered_data = np.clip(self.sigma * np.random.randn(N, C), -1 * self.clip, self.clip)
        jittered_data += points
        return jittered_data


class select_point_cloud(object):
    def __init__(self, n_pts=2048):
        self.n_pts = n_pts

    def __call__(self, points):
        choice = random.sample(range(0, 2048), self.n_pts)
        points = points[choice, :]
        return points


class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (n_pts x C) to a list of torch.FloatTensor of shape (C x n_pts)"""

    def __call__(self, points):
        points = np.transpose(points, (1, 0))
        return points
