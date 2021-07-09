import gudhi
import logging
import numpy as np
from scipy.spatial.distance import cdist, pdist
from tqdm import tqdm
from itertools import permutations, combinations
import matplotlib.pyplot as plt
from collections import deque
from time import time
import ot
from TDA.wasserstein import wasserstein_distance
from sklearn.neighbors import KDTree


def MatrixNorm(Kernel, norm_type, diff_dist=False, **kwargs):
    if isinstance(Kernel, float):
        return Kernel

    # calculate diffusion distance matrix
    mat = Kernel
    return np.linalg.norm(mat, norm_type)


def Inverse(w, **kwargs):
    return 1 / w


def ExponentialInverse(w, **kwargs):
    return 1 / np.exp(w)


def NormalizeInverse(w, **kwargs):
    root_deg = kwargs.get('root_deg')
    return (1 / w) ** (1 / root_deg)


def LogInverse(w, **kwargs):
    return np.log(1 + (1 / w))


def Identity(w, **kwargs):
    return w


def WeightFuncSelection(WeightFunc):
    # set relation weight function
    if WeightFunc == 'Inverse':
        WeightHandle = Inverse
    elif WeightFunc == 'NormalizeInverse':
        WeightHandle = NormalizeInverse
    elif WeightFunc == 'LogInverse':
        WeightHandle = LogInverse
    elif WeightFunc == 'Identity':
        WeightHandle = Identity
    elif WeightFunc == 'ExponentialInverse':
        WeightHandle = ExponentialInverse

    return WeightHandle


def KernelMultiplication(Kernels, Method='symmetric', **kwargs):
    """
    Kernel multiplication for simplex edge and face relations - Alternation Diffusion
    :param Kernels: list of kernels. 2 kernels in edge 3 fro face etc..
    :param Symmetric: state if the relations will be symmetric meaning in both ways
    :return: Alternating diffusion kernel
    """

    NumOfKernel = len(Kernels)
    K = np.zeros(Kernels[0].shape)
    if NumOfKernel == 1:
        return Kernels

    if Method == 'symmetric':
        AllPermutations = list(permutations(range(NumOfKernel)))
        NumOfPermute = len(AllPermutations)
        for KernelInd in AllPermutations:
            auxK = Kernels[KernelInd[0]]
            for ii in KernelInd[1:]:
                auxK = auxK @ Kernels[ii]  # (K_1 @ K_2 @ .... @ K_n) + (K_n @ K_(n-1) @ .... @ K_1)
            K += auxK
        K = K / NumOfPermute
    elif Method == 'direct':
        # not Symmetric relation
        K = Kernels[0]
        for auxK in Kernels[1:]:
            # K = K_1 * K_2 * K_3 * .... *K_n
            K = K @ auxK
        # K = K@K  # Alternating of 2
    elif Method == 'eigenvalues':
        KernelEigenvalues = [np.linalg.eig(Kernel)[0] for Kernel in Kernels]
        AllCombinations = list(combinations(range(NumOfKernel), 2))
        K = 0
        for KernelInd in AllCombinations:
            K += np.linalg.norm((KernelEigenvalues[KernelInd[0]] - KernelEigenvalues[KernelInd[1]]))
    elif Method == 'addition':
        AllPermutations = list(permutations(range(NumOfKernel)))
        NumOfPermute = len(AllPermutations)
        for KernelInd in AllPermutations:
            auxK = Kernels[KernelInd[0]]
            for ii in KernelInd[1:]:
                # auxK = auxK.dot(Kernels[ii].transpose())  # (K_1 @ K_2 @ .... @ K_n) + (K_n @ K_(n-1) @ .... @ K_1)
                auxK = auxK @ Kernels[ii].T  # (K_1 @ K_2 @ .... @ K_n) + (K_n @ K_(n-1) @ .... @ K_1)
            K += auxK
        K = K / NumOfPermute
    elif Method == 'subtraction':
        K = Kernels[0] @ Kernels[1].T - Kernels[1] @ Kernels[0].T

    return K


def GetDiagramIntervals(diag, dim):
    """
    :param diag: diagram for intravals
    :param dim: the required dimension
    :return: the intravals
    """
    Intervals = [diag[ii][1] for ii in range(len(diag)) if diag[ii][0] == dim]
    return Intervals


def get_kernel_regularization_func(kernel_regularization, eps):
    if kernel_regularization:
        def kernel_reg_cal(loc):
            D = np.concatenate((pdist(np.expand_dims(loc[:, 0], axis=1)), pdist(np.expand_dims(loc[:, 1], axis=1))))
            return np.exp((np.max(D) - 1) / eps)
        return kernel_reg_cal
    return lambda loc: 1


def PersistentDiagram(Kernels,
                      valid_kernels_idx,
                      KernelNorm='fro',
                      WeightFunc='Inverse',
                      WeighingMethod='addition',
                      Diff_Dist=False,
                      **kwargs):
    """
    PersistentDiagram calculate persistent diagram for given list of kernels
    :param SemiticKernel: stating of the relation over edge is semitric
    :param Kernels: list of kernels
    :param WeightFunc: type of weight function for evaluation the relation between kernels in simplex
    :return:
    """
    Plot = kwargs.get('Plot', False)
    kernel_regularization = kwargs.get('kernel_regularization', False)
    kernel_regularization_eps = kwargs.get('kernel_regularization_eps', 1)
    kernel_location = kwargs.get('kernel_location', np.zeros(len(Kernels)))
    low_of_three = kwargs.get('low_of_three', False)

    weights_edges = np.zeros((len(Kernels), len(Kernels)))
    weights_edges_dict = {}
    weights_faces_dict = {}

    n_kernels = len(valid_kernels_idx)
    WeightHandle = WeightFuncSelection(WeightFunc)
    K_size = Kernels[valid_kernels_idx[0]].shape
    kernel_regularization_func = get_kernel_regularization_func(kernel_regularization, kernel_regularization_eps)

    # calculate and insert edges weights to simplex
    Edge_Operator_dict = {}
    EdgeCombinations = combinations(valid_kernels_idx, 2)
    for ind_key in EdgeCombinations:
        EdgeOperator = KernelMultiplication([Kernels[ind_key[0]],
                                             Kernels[ind_key[1]]],
                                            Method=WeighingMethod)
        EdgeOperatorNorm = MatrixNorm(EdgeOperator, KernelNorm, diff_dist=Diff_Dist)
        Edge_Operator_dict[ind_key] = EdgeOperator
        weights_edges_dict[ind_key] = WeightHandle(EdgeOperatorNorm, root_deg=2)

    # calculate and insert Face(simplex) weights to simplex
    FaceCombinations = list(combinations(valid_kernels_idx, 3))

    for ind_key in FaceCombinations:
        kernels_indices = deque(ind_key)

        # calculate the weights of edges and faces
        FaceOperator = np.zeros(K_size)
        for _ in range(len(kernels_indices)):
            Edge_key = tuple(np.sort([kernels_indices[0], kernels_indices[1]]))
            EdgeOperator = Edge_Operator_dict[Edge_key]
            FaceOperator += KernelMultiplication([EdgeOperator,
                                                  Kernels[kernels_indices[2]]],
                                                 Method=WeighingMethod)
            kernels_indices.rotate(1)

        FaceOperator = FaceOperator / 3  # uniformity the sum of 3 kernels
        FaceOperatorNormSimplex = MatrixNorm(FaceOperator, KernelNorm, diff_dist=Diff_Dist)
        weights_faces_dict[ind_key] = WeightHandle(FaceOperatorNormSimplex, root_deg=3)

    kernel_location_vec = np.full(len(Kernels), -1)
    kernel_location_vec[valid_kernels_idx] = np.arange(n_kernels)

    # build simplex tree
    Diag = BuildSimplicialComplex(n_kernels,
                                  weights_edges_dict,
                                  weights_faces_dict,
                                  low_of_three=low_of_three,
                                  vertices_location=kernel_location_vec,
                                  plot=Plot)

    if Plot:
        edge_val = np.fromiter(weights_edges_dict.values(), dtype=float)
        face_val = np.fromiter(weights_faces_dict.values(), dtype=float)
        plt.figure()
        plt.imshow(weights_edges)
        plt.title("Edges weights")
        plt.colorbar()

        plt.figure()
        plt.hist([edge_val, face_val], bins=100)
        plt.hist([edge_val], bins=100)
        plt.hist([face_val], bins=100)
        plt.figure()
        plt.plot(np.sort(edge_val), 'g', marker='o')
        plt.plot(np.sort(face_val), 'r', marker='o')
        plt.draw()

    return Diag


def PersistentNeighborsDiagram(Kernels,
                               valid_kernels_idx,
                               node_neighbors_vec,
                               KernelNorm='fro',
                               WeightFunc='Inverse',
                               WeighingMethod='addition',
                               Diff_Dist=False,
                               **kwargs):
    """
    PersistentDiagram calculate persistent diagram for given list of kernels and compute the relative neighbors
    :param Kernels: list of kernels
    :param WeightFunc: type of weight function for evaluation the relation between kernels in simplex
    :return:
    """
    Plot = kwargs.get('Plot', False)
    kernel_regularization = kwargs.get('kernel_regularization', False)
    kernel_regularization_eps = kwargs.get('kernel_regularization_eps', 1)
    kernel_location = kwargs.get('kernel_location')
    low_of_three = kwargs.get('low_of_three', False)

    weights_edges_dict = {}
    weights_faces_dict = {}
    WeightHandle = WeightFuncSelection(WeightFunc)
    kernel_regularization_func = get_kernel_regularization_func(kernel_regularization, kernel_regularization_eps)

    n_kernels = valid_kernels_idx.shape[0]
    K_size = Kernels[valid_kernels_idx[0]].shape

    # calculate and insert edges weights to simplex
    Edge_Operator_dict = {}
    for node_idx, neighbors in zip(valid_kernels_idx, node_neighbors_vec):
        neighbors = neighbors['edges']
        for edge_neighbor_idx in neighbors:
            two_neighbors = np.sort([node_idx, edge_neighbor_idx])
            two_neighbors = (two_neighbors[0], two_neighbors[1])
            if two_neighbors not in weights_edges_dict.keys():
                EdgeOperator = KernelMultiplication([Kernels[two_neighbors[0]],
                                                     Kernels[two_neighbors[1]]],
                                                    Method=WeighingMethod)
                Edge_Operator_dict[two_neighbors] = EdgeOperator
                EdgeOperatorNorm = MatrixNorm(EdgeOperator, KernelNorm, diff_dist=Diff_Dist)
                weights_edges_dict[two_neighbors] = WeightHandle(EdgeOperatorNorm, root_deg=2)

    # calculate and insert Face(simplex) weights to simplex


    for node_idx, neighbors in zip(valid_kernels_idx, node_neighbors_vec):
        neighbors = neighbors['faces']
        for face_neighbors in neighbors:
            if face_neighbors is not None:
                face_neighbors = tuple([idx for idx in np.sort(face_neighbors)])
                three_neighbors = np.sort([node_idx, face_neighbors[0], face_neighbors[1]])
                three_neighbors = (three_neighbors[0], three_neighbors[1], three_neighbors[2])
                if (three_neighbors not in weights_faces_dict.keys()) and (face_neighbors in weights_edges_dict.keys()):
                    kernels_indices = deque(three_neighbors)
                    # calculate the weights of edges and faces
                    FaceOperator = np.zeros(K_size)
                    for _ in range(len(kernels_indices)):
                        Edge_key = tuple(np.sort([kernels_indices[0], kernels_indices[1]]))
                        EdgeOperator = Edge_Operator_dict[Edge_key]
                        FaceOperator += KernelMultiplication([EdgeOperator,
                                                              Kernels[kernels_indices[2]]],
                                                             Method=WeighingMethod)
                        kernels_indices.rotate(1)

                    FaceOperator = FaceOperator / 3  # uniformity the sum of 3 kernels
                    FaceOperatorNormSimplex = MatrixNorm(FaceOperator, KernelNorm, diff_dist=Diff_Dist)
                    weights_faces_dict[three_neighbors] = WeightHandle(FaceOperatorNormSimplex, root_deg=3)


    kernel_location_vec = np.full(len(Kernels), -1)
    kernel_location_vec[valid_kernels_idx] = np.arange(n_kernels)

    Diag = BuildSimplicialComplex(n_kernels,
                                  weights_edges_dict,
                                  weights_faces_dict,
                                  low_of_three=low_of_three,
                                  vertices_location=kernel_location_vec,
                                  plot=Plot)

    if Plot:
        edge_val = np.fromiter(weights_edges_dict.values(), dtype=float)
        face_val = np.fromiter(weights_faces_dict.values(), dtype=float)

        plt.figure()
        plt.hist([edge_val, face_val], bins=100)
        plt.hist([edge_val], bins=100)
        plt.hist([face_val], bins=100)
        plt.figure()
        plt.plot(np.sort(edge_val), 'g', marker='o')
        plt.plot(np.sort(face_val), 'r', marker='o')
        plt.legend(('edges', 'faces'))
        plt.draw()

    return Diag


def BuildSimplicialComplex(n_vertices,
                           edge_weights_dict,
                           face_weights_dict,
                           **kwargs):
    low_of_three = kwargs.get('low_of_three', False)
    vertices_location = kwargs.get('vertices_location', np.arange(n_vertices))
    SimplexTree = gudhi.SimplexTree()

    # set vertices: init vertices with 0 value for filtration
    for ii in range(n_vertices):
        SimplexTree.insert([ii],
                           filtration=0.0)
    # set edges:
    for ind_key in edge_weights_dict.keys():
        EdgeWeight = edge_weights_dict[ind_key]
        SimplexTree.insert([vertices_location[ind_key[0]],
                            vertices_location[ind_key[1]]],
                           filtration=EdgeWeight)

    # set faces:
    count_faces = 0
    for ind_key in face_weights_dict.keys():
        FaceWeight = face_weights_dict[ind_key]
        for edge_idx in combinations(ind_key, 2):
            if FaceWeight < edge_weights_dict[edge_idx]:
                logging.debug("Face lower over vertices: %s" % (ind_key,))
                count_faces += 1
                if low_of_three:
                    FaceWeight = edge_weights_dict[edge_idx]
        SimplexTree.insert([vertices_location[ind_key[0]],
                            vertices_location[ind_key[1]],
                            vertices_location[ind_key[2]]],
                           filtration=FaceWeight)
    try:
        logging.info("%.2f lower faces weights - # %d faces" % (count_faces/len(face_weights_dict), len(face_weights_dict)))
    except:
        logging.info("no faces faces weights - # faces")



    SimplexTree.initialize_filtration()
    Diag = SimplexTree.persistence()

    if kwargs.get('plot', False):
        plt.figure()
        plt.subplot(1, 2, 1)
        gudhi.plot_persistence_barcode(Diag)
        plt.subplot(1, 2, 2)
        gudhi.plot_persistence_diagram(Diag)

    return Diag


def BuildClickComplex(n_vertices,
                      edge_weights_dict,
                      **kwargs):
    edges_set = edge_weights_dict.keys()
    SimplexTree = gudhi.SimplexTree()
    for idx in range(n_vertices):
        SimplexTree.insert([idx],
                           filtration=0.0)
    # set edges:
    for idx_key in edges_set:
        EdgeWeight = edge_weights_dict[idx_key]
        SimplexTree.insert([idx_key[0], idx_key[1]],
                           filtration=EdgeWeight)
    # set faces:
    for idx in range(n_vertices):
        FaceWeight = np.max([weight_mat[idx] for idx in list(combinations(ind_key, 2))])
        FaceWeight = np.max([FaceWeight, 0])

        # assert FaceEdgeWeightCompare(FaceWeight, weights_edges, ind_key), \
        #     "Face weight is smaller then edge weight when building graph"
        SimplexTree.insert([ind_key[0], ind_key[1], ind_key[2]],
                           filtration=FaceWeight)

    SimplexTree.initialize_filtration()
    Diag = SimplexTree.persistence()

    if kwargs.get('plot', False):
        plt.figure()
        plt.subplot(1, 2, 1)
        gudhi.plot_persistence_barcode(Diag)
        plt.subplot(1, 2, 2)
        gudhi.plot_persistence_diagram(Diag)


def CovarianceNeighborsClickComplex(samples,
                                    valid_kernels_idx,
                                    node_neighbors_vec,
                                    KernelNorm='fro',
                                    WeightFunc='Inverse',
                                    WeighingMethod='addition',
                                    normalize=False,
                                    Diff_Dist=False,
                                    **kwargs):
    samples_std = np.diag(1 / np.std(samples, axis=1))
    weight_mat = samples_std @ np.cov(samples) @ samples_std
    weight_mat = weight_mat.max() - np.abs(weight_mat)

    n_vertices = samples.shape[0]
    weights_edges_dict = {}
    weights_faces_dict = {}

    # calculate and insert edges weights to simplex

    # set edges:
    for node_idx, neighbors in zip(valid_kernels_idx, node_neighbors_vec):
        neighbors = neighbors['edges']
        for edge_neighbor_idx in neighbors:
            two_neighbors = np.sort([node_idx, edge_neighbor_idx])
            two_neighbors = (two_neighbors[0], two_neighbors[1])
            if two_neighbors not in weights_edges_dict.keys():
                weights_edges_dict[two_neighbors] = weight_mat[two_neighbors[0], two_neighbors[1]]

    # calculate and insert Face(simplex) weights to simplex
    # Click Complex - largest then three edges

    for node_idx, neighbors in zip(valid_kernels_idx, node_neighbors_vec):
        neighbors = neighbors['faces']
        for face_neighbors in neighbors:
            if face_neighbors is not None:
                face_neighbors = tuple([idx for idx in np.sort(face_neighbors)])
                three_neighbors = np.sort([node_idx, face_neighbors[0], face_neighbors[1]])
                three_neighbors = (three_neighbors[0], three_neighbors[1], three_neighbors[2])
                if (three_neighbors not in weights_faces_dict.keys()) and (face_neighbors in weights_edges_dict.keys()):
                    kernels_indices = deque(three_neighbors)
                    # calculate the weights of edges and faces
                    FaceWieght = 0
                    for _ in range(len(kernels_indices)):
                        Edge_key = tuple(np.sort([kernels_indices[0], kernels_indices[1]]))
                        edge_weight = weight_mat[Edge_key[0], Edge_key[1]]
                        FaceWieght = edge_weight if edge_weight > FaceWieght else FaceWieght
                        kernels_indices.rotate(1)

                    weights_faces_dict[three_neighbors] = FaceWieght

    Diag = BuildSimplicialComplex(n_vertices,
                                  weights_edges_dict,
                                  weights_faces_dict,
                                  plot=False)

    return Diag


def DiagramDistance(DiagList, Dimension, distance='bottleneck', clean_small_homology=None):
    """
    DiagramDistance returns distance matrix between diagrams in the given list for given dimension
    :param DiagList:
    :param Dimension:
    :return:
    """
    # TODO: enhance performance by using scipy.pdist
    if distance == 'bottleneck':
        def distance_func(a, b, dimension):
            a = [interval for interval in a if interval[1] != float('inf')]
            b = [interval for interval in b if interval[1] != float('inf')]
            return gudhi.bottleneck_distance(a, b)
    elif distance == 'wasserstein':
        def distance_func(a, b, dimension, p=2):
            # clear inf interval (things that don't die)
            a, b = np.array(a), np.array(b)
            try:
                a = a[a[:, 1] != np.inf, :]
                b = b[b[:, 1] != np.inf, :]
            except:
                return 0

            return wasserstein_distance(a, b)


    # pre-processing
    clean_diag_list = []
    for diag in DiagList:
        diag = GetDiagramIntervals(diag, Dimension)
        if clean_small_homology is not None:
            diag_aux = np.array(diag)
            valid_idx = np.argsort((diag_aux[:, 1] - diag_aux[:, 0]))[::-1]
            diag = [diag[ii] for ii in valid_idx[0: int(len(diag) * clean_small_homology)]]
        clean_diag_list.append(diag)

    NumOfAttributes = len(DiagList)
    matDiagDist = np.zeros([NumOfAttributes, NumOfAttributes])
    for DiagInd in tqdm(range(NumOfAttributes)):
        diag_A = clean_diag_list[DiagInd]
        A_betti = np.sum(np.array(diag_A)[:, 1] == np.inf)

        for CompareDiagInd in range(DiagInd + 1, NumOfAttributes):
            diag_B = clean_diag_list[CompareDiagInd]
            B_betti = np.sum(np.array(diag_B)[:, 1] == np.inf)

            if A_betti != B_betti:
                logging.info("Diagram distance: betti number arn't equal!!!!!!")
            Dist = distance_func(diag_A, diag_B, Dimension)

            # symmetric matrix
            matDiagDist[DiagInd, CompareDiagInd] = Dist
            matDiagDist[CompareDiagInd, DiagInd] = Dist

    return matDiagDist


def FaceEdgeWeightCompare(face_weights, edge_mat, idx):
    for idx_key in list(combinations(idx, 2)):
        if face_weights < edge_mat[idx_key]:
            return True
    return False
