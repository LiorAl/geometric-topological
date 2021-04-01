import time
import logging
import numpy as np
import json
from scipy.interpolate import interp1d
from sklearn.neighbors import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull
import os


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name)
        print('Elapsed: %s' % (time.time() - self.tstart))


def PrepareResultsDir(WorkDir,
                      base_name="",
                      debug=False,
                      figures=False):
    if debug:
        # check if running in debug mod
        result_path = os.getcwd() + '/Results/Debug'
        return result_path
    elif figures:
        result_path = os.getcwd() + '/Results/Figures'
        return result_path

    # Create results directory
    result_path = os.getcwd() + '/' + WorkDir + '/' + base_name + time.strftime('%b_%d_%H_%M_%S', time.localtime())
    os.mkdir(result_path)
    local_dir = os.path.dirname(__file__)
    # for py_file in glob.glob(local_dir + '/*.py'):
    #     copy(py_file, result_path)

    return result_path


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path, **kwargs):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        json.dump(d, f, indent=4)


def im2col(input_data, filter_h, filter_w, stride=1, pad=(0, 0)):
    img = np.pad(input_data, [(0, 0), (0, 0), (pad[0], pad[0]), (pad[1], pad[1])], 'edge')
    N, C, H, W = img.shape
    NN, CC, HH, WW = img.strides
    out_h = (H - filter_h)//stride + 1
    out_w = (W - filter_w)//stride + 1
    col = np.lib.stride_tricks.as_strided(img,
                                          (N, out_h, out_w, C, filter_h, filter_w),
                                          (NN, stride * HH, stride * WW, CC, HH, WW)).astype(float)
    return col.reshape(np.multiply.reduceat(col.shape, (0, 1, 3))), (out_h, out_w)


def get_vertex_neighbors(grid_size, valid_patches_idx_vec):
    n_rows, n_columns = grid_size[0], grid_size[1]
    n_vertex = n_rows * n_columns
    vertex_idx_vec = np.array(np.unravel_index(np.arange(n_vertex), grid_size)).T

    def edge_neighbors_func(ii, jj):
        edge_neighbors = np.array([[ii - 1, jj - 1], [ii - 1, jj],
                                   [ii, jj - 1], [ii, jj + 1],
                                   [ii + 1, jj], [ii + 1, jj + 1]])
        face_neighbors = np.array([[[ii - 1, jj - 1], [ii - 1, jj]],
                                   [[ii - 1, jj - 1], [ii, jj - 1]],
                                   [[ii, jj + 1], [ii + 1, jj + 1]],
                                   [[ii + 1, jj], [ii + 1, jj + 1]]])

        # check if vertex is out of grid
        edge_neighbors = edge_neighbors[np.all(edge_neighbors >= 0, axis=1)]
        edge_neighbors = edge_neighbors[edge_neighbors[:, 0] < n_rows]
        edge_neighbors = edge_neighbors[edge_neighbors[:, 1] < n_columns]

        face_neighbors = face_neighbors[np.all(face_neighbors.reshape(face_neighbors.shape[0], -1) >= 0, axis=1)]
        face_neighbors = face_neighbors[np.all(face_neighbors[:, :, 0] < n_rows, axis=1)]
        face_neighbors = face_neighbors[np.all(face_neighbors[:, :, 1] < n_columns, axis=1)]

        # convert to linear indices
        edge_neighbors = np.ravel_multi_index(edge_neighbors.T, grid_size)
        face_neighbors = np.array([np.ravel_multi_index(edges.T, grid_size)
                                   for edges in face_neighbors])

        # get intersect with valid vertex:
        edge_neighbors = np.intersect1d(edge_neighbors, valid_patches_idx_vec, assume_unique=True)
        face_neighbors = np.array([np.intersect1d(neighbors, valid_patches_idx_vec, assume_unique=True)
                                   for neighbors in face_neighbors])
        # clean non valid face because single neighbor
        face_neighbors = np.array([neighbors if len(neighbors) > 1 else None
                                   for neighbors in face_neighbors])

        return {'edges' : edge_neighbors,
                'faces' : face_neighbors}

    vertex_neighbors_list = list(map(edge_neighbors_func,
                                     list(vertex_idx_vec[:, 0]),
                                     list(vertex_idx_vec[:, 1])))
    # clean non-valid patches
    vertex_neighbors_list = [vertex_neighbors_list[idx] for idx in valid_patches_idx_vec]
    return vertex_neighbors_list


def split_image(input_data, stride=1, pad=(0, 0)):
    H, W, _ = input_data.shape
    filter_h, filter_w = H//2, W//2
    x1 = input_data[0:filter_h, 0:filter_w, :]
    x2 = input_data[0:filter_h, filter_w:, :]
    x3 = input_data[filter_h:, 0:filter_w, :]
    x4 = input_data[filter_h:, filter_w:, :]
    return x1, x2, x3, x4


def get_valid_patches(patches, grid_size, patch_size=35, std_threshold=1/3):

    n_patches = patches.shape[1]

    # std threshold for patches
    std_patches = np.std(patches, axis=-1)
    mean_std_patches = np.mean(std_patches, axis=0).reshape(grid_size)
    valid_patches = mean_std_patches >= mean_std_patches.max() * std_threshold

    #  if not all patches are valid
    if np.sum(valid_patches) != n_patches:
        # patch location
        non_valid_patches = np.logical_not(valid_patches)
        valid_patch_location = np.array(np.nonzero(valid_patches)).T
        non_valid_patch_location = np.array(np.nonzero(non_valid_patches)).T
        valid_patches = valid_patches.flatten()

        # generate convex hull
        hull = ConvexHull(valid_patch_location)

        def point_in_hull(point, tolerance=1e-12):
            return all(
                (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
                for eq in hull.equations)

        add_convex_vertices = list(map(point_in_hull, non_valid_patch_location))
        valid_patch_location = np.vstack((valid_patch_location, non_valid_patch_location[add_convex_vertices, :]))
        non_valid_patch_location = non_valid_patch_location[np.logical_not(add_convex_vertices), :]


        # search tree
        neighbors_tree = KDTree(valid_patch_location)
        node_neighbors_vec = neighbors_tree.query_radius(valid_patch_location, r=1.5, count_only=True)
        # clear patches less then 4 neighbors and add patches with 7 neigbores
        node_neighbors_valid_loc = node_neighbors_vec > 3
        valid_patches[np.ravel_multi_index(valid_patch_location.T, grid_size)] = node_neighbors_valid_loc

        if len(non_valid_patch_location):
            node_with_full_neighbors = neighbors_tree.query_radius(non_valid_patch_location, r=1.5, count_only=True)
            node_neighbors_valid_loc = node_with_full_neighbors > 7
            valid_patches[np.ravel_multi_index(non_valid_patch_location.T, grid_size)] = node_neighbors_valid_loc

        # update tree with valid neighbors
        valid_patch_location = np.array(np.nonzero(valid_patches.reshape(grid_size))).T
        neighbors_tree = KDTree(valid_patch_location)
        node_neighbors_vec = neighbors_tree.query_radius(valid_patch_location, r=1)

        # build graph for connected components analysis
        adjacency_mat = np.zeros((n_patches, n_patches))
        for node_idx, node_neighbors in enumerate(node_neighbors_vec):
            valid_patch_idx = np.ravel_multi_index(valid_patch_location[node_idx], grid_size)
            node_neighbors_idx = np.ravel_multi_index(valid_patch_location[node_neighbors].T, grid_size)
            node_neighbors_idx = np.setdiff1d(node_neighbors_idx, valid_patch_idx)
            adjacency_mat[valid_patch_idx, node_neighbors_idx] = 1

        # find the largest connected component
        graph = csr_matrix(adjacency_mat)
        n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        counts = np.bincount(labels)
        largest_components = labels == np.argmax(counts)

        # update valid patches
        valid_patches = valid_patches & largest_components

    return np.array(np.nonzero(valid_patches))

