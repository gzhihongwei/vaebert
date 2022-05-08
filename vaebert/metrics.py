import numpy as np
import torch

from scipy.spatial import KDTree


def chamfer_dist(x, y):
    # convert x and y to binary (integer) torch tensor
    x = x.cpu().numpy().squeeze().round().astype(np.int8)
    y = y.cpu().numpy().squeeze().round().astype(np.int8)

    # create kd-tree for x and y
    x_tree = KDTree(np.transpose(x.nonzero()))
    y_tree = KDTree(np.transpose(y.nonzero()))

    # we only care about the set of points that are in one point cloud but not the other
    # since if a point is in both clouds their closest distance is 0

    # get the set of points common to both point clouds
    common = x * y

    # get nonzero coordinates of unique points in both clouds
    unique_x = np.transpose((x - common).nonzero())
    unique_y = np.transpose((y - common).nonzero())

    # for each unique point in one cloud, find the closest point in the other
    # and vice versa
    d_x, _ = y_tree.query(unique_x, k=1)
    d_y, _ = x_tree.query(unique_y, k=1)

    return d_x.sum() / (x.sum() + np.finfo(float).eps) + d_y.sum() / (
        y.sum() + np.finfo(float).eps
    )
