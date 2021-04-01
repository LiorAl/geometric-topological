from scipy.spatial.distance import cdist, pdist, squareform
from scipy.signal import stft, spectrogram
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import ot
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


"""
Alternating-Diffusion algorithm function list
"""

def PairWiseDistance(S, Distance, **kwargs):
    # pairwise distance between samples ||s_i - s_j|| - matrix that the cell (i,j) is
    #  the euclidean distance between observation i and j
    # input: S1 - samples from first source
    #        S2 - samples from second source
    #        Distance - type of distance, default 'euclidean'
    patch_size = kwargs.get('patch_size')
    scale_samples = kwargs.get('normalize_scale', False)
    if Distance == 'seuclidean':
        S = np.copy((S.T - np.mean(S, axis=1)).T)
        var = np.var(S, axis=0) + 1e-9
        Dist = cdist(S, S, metric='seuclidean', V=var) ** 2

    if Distance == 'EMD':
        vertex_location = np.array(np.unravel_index(np.arange(S.shape[1]), (patch_size, patch_size))).T
        vertex_distance = squareform(pdist(vertex_location))
        vertex_distance /= vertex_distance.max()

        def EarthMoversDistance(patch1, patch2):
            """
             EarthMovingDistance calculate the EMD distance between two patches of images, including calculating
             the cost matrix (simple pairwise euclidean distance)
             """
            # all distribution must be positive
            patch1_bias = 2 * patch1.min() if patch1.min() < 0 else 0
            patch2_bias = 2 * patch2.min() if patch2.min() < 0 else 0
            patch1 = (patch1 - patch1_bias) / np.sum(patch1 - patch1_bias)
            patch2 = (patch2 - patch2_bias) / np.sum(patch2 - patch2_bias)
            return ot.emd2(patch1, patch2, vertex_distance)


        Dist = cdist(S, S, EarthMoversDistance)


    else:
        if scale_samples:
            scaler = StandardScaler()
            scaler.fit(S.T)
            S = scaler.transform(S.T).T
        Dist = cdist(S, S, 'sqeuclidean')

    return Dist


def AffinityKernels(DistanceMat, epsilon_factor=1, uniformity=True):
    # AffinityKernels Computes an affinity matrix from given distances
    # epsilon - is the median of all distance

    #  input: DistanceMat - pairwise distance, must be square and symmetric

    epsilon = np.median(DistanceMat) * epsilon_factor + 1e-9
    W = np.exp(- (1 / epsilon) * DistanceMat)
    D = np.diag(1 / np.sum(W, axis=1))
    if uniformity:
        W = D @ W @ D
    return W


def DiffusionKernel(AffinityMat):
    #  DiffusionKernel generate a diffusion kernel K for diffusion
    #    maps, from a given symmetric affinity matrix AffinityMat by the maps algorithm, 2 times uniformity

    RowDim = 1
    ColumnSum = np.sum(AffinityMat, RowDim)  # for uniformity
    # Row normalization
    Kernel = np.diag(1 / ColumnSum) @ AffinityMat

    return Kernel


def DiffusionDistance(Diffusion_Operator):
    return cdist(Diffusion_Operator, Diffusion_Operator, lambda u, v: np.sum((u - v) ** 2))


def DiffusionMap(W):
    # U, V, _ = np.linalg.svd(K)
    D = np.diag(np.sum(W, axis=1) ** (- 1 / 2))
    W_normalized = D @ W @ D
    Lambda_vec, Phi_mat = np.linalg.eig(W_normalized)
    sort_idx = np.argsort(Lambda_vec)
    sort_idx = sort_idx[::-1]  # decent order
    Lambda_vec = Lambda_vec[sort_idx]
    Phi_mat = Phi_mat[:, sort_idx]
    PhiK = np.diag(1 / Phi_mat[:, 0]) @ Phi_mat
    MapEmbedding = PhiK @ np.diag(Lambda_vec)
    MapEmbedding = MapEmbedding[:, 1:]
    DiffusionDist = PairWiseDistance(MapEmbedding, 'euclidean')
    return MapEmbedding, DiffusionDist, Lambda_vec[1:]


def GetDiffusionMapsKernel(matSamples, epsilon_factor=1, metric='euclidean', custom_func=None, **kwargs):
    """

    :param matSamples:
    :param epsilon_factor:
    :param metric: can be standard metric or 'custom' then custom_func should be supplied
    :param custom_func:
    :return:
    """
    # applying alternating Diffusion
    # Distance between samples
    try:
        affinity_threshold = kwargs.get('affinity_threshold')
    except:
        affinity_threshold = None
    if metric == 'custom':
        Dist = custom_func(matSamples, kwargs.get('radii_set'))
    else:
        Dist = PairWiseDistance(matSamples, metric)

    # Affinity matrices
    uniformity = kwargs.get('uniformity', False)
    W = AffinityKernels(Dist,
                        epsilon_factor=epsilon_factor,
                        uniformity=uniformity)

    # Diffusion kernel
    K = DiffusionKernel(W)

    return K


def SingleSensorDiffusionKernel(samples,
                                lag=10,
                                sample_preprocess=None,
                                kernel_distance='euclidean',
                                lag_overlap=None,
                                **kwargs):
    """
    This function build lag-step series to given attribute of sensor and return alternating diffusion kernel
    :return: Kernel - the AD kernel
    """
    patch_size      = kwargs.get('patch_size')
    epsilon_factor  = kwargs.get('epsilon_factor', 1)
    normalize_scale = kwargs.get('normalize_scale', False)

    if sample_preprocess == 'frequency':
        fs, freq_range = kwargs.get('fs'), kwargs.get('freq_range')
        f, t, S = stft(samples, fs, nperseg=lag, noverlap=lag_overlap, padded=False)
        if freq_range:
            min_freq_idx = (np.abs(f - freq_range[0])).argmin()
            max_freq_idx = (np.abs(f - freq_range[1])).argmin()
            f_idx = np.arange(min_freq_idx, max_freq_idx)
        else:
            f_idx = np.arange(0, len(f))
        S = np.abs(S[f_idx, :]).T

    elif sample_preprocess == 'lagmap':
        S = [samples[ii: ii + lag] for ii in range(0, samples.shape[0] - lag, lag - lag_overlap)]
    else:
        S = samples

    # Distance between samples
    if sample_preprocess == 'covariance':
        Dist = np.cov(samples)
        K = DiffusionKernel(Dist)

    else:
        Dist = PairWiseDistance(S, kernel_distance,
                                patch_size=patch_size,
                                normalize_scale=normalize_scale)
        # Affinity matrices
        uniformity = kwargs.get('uniformity', False)
        W = AffinityKernels(Dist, uniformity=uniformity, epsilon_factor=epsilon_factor)
        # Diffusion kernel
        K = DiffusionKernel(W)

    ###### for plot
    if kwargs.get('plot', False):
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(Dist)
        plt.title('Dist')
        plt.colorbar()
        plt.subplot(1, 3, 2)
        plt.imshow(W)
        plt.title('W')
        plt.colorbar()
        plt.subplot(1, 3, 3)
        plt.imshow(K)
        plt.title('K')
        plt.colorbar()

    return K


def SensorsDiffusionKernel(samples,
                           lag=10,
                           sample_preprocess=None,
                           kernel_distance='euclidean',
                           lag_overlap=None,
                           **kwargs):
    """
    SensorsDiffusionKernel build build kernels for lag in time, which each kernel is the build on
    similarity between sensors
    :return: kernel_list - list of kernel for each time lag
    """

    n_sensors = samples.shape[1]
    n_samples = samples.shape[0]
    # build lag idx vector
    if sample_preprocess == 'cross-correlation':
        n_kernels = n_sensors
    else:
        lag_vec = np.arange(0, n_samples - lag, lag - lag_overlap)
        n_kernels = len(lag_vec)

    kernels_list = [None] * n_kernels

    for kernel_idx in range(n_kernels):
        if sample_preprocess == 'frequency':
            freq_overlap = 8
            fs, freq_range = kwargs.get('fs'), kwargs.get('freq_range')
            Sxx_mat = []
            for sensors_idx in range(0, n_sensors):
                f, t, Sxx = stft(samples[lag_vec[kernel_idx]: lag_vec[kernel_idx] + lag, sensors_idx],
                                 fs,
                                 nperseg=lag,
                                 noverlap=lag - freq_overlap,
                                 padded=False)
                if freq_range:
                    min_freq_idx = (np.abs(f - freq_range[0])).argmin()
                    max_freq_idx = (np.abs(f - freq_range[1])).argmin()
                    f_idx = np.arange(min_freq_idx, max_freq_idx)
                else:
                    f_idx = np.arange(0, len(f))
                Sxx = np.abs(Sxx[f_idx, :]).T
                Sxx_mat.append(np.mean(Sxx, axis=0))
            S = np.array(Sxx_mat)

        if sample_preprocess == 'cross-correlation':
            S = [None] * n_sensors
            for sensors_idx in range(n_sensors):
                S[sensors_idx] = np.correlate(samples[:, kernel_idx], samples[:, sensors_idx], mode='full') / n_samples
            S = np.array(S)

            # Distance between samples
        Dist = PairWiseDistance(S, kernel_distance)
        # Affinity matrices
        W = AffinityKernels(Dist, uniformity=True)
        # Check if need to uniform the affinity matrix

        # Diffusion kernel
        K = DiffusionKernel(W)

        kernels_list[kernel_idx] = K

    return kernels_list
