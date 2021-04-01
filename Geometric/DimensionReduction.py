import AlternatingDiffusion as AD
from sklearn.manifold import TSNE, MDS, SpectralEmbedding
import matplotlib.pyplot as plt
import phate.phate as phate
import numpy as np
import logging


def DiffusionMaps(D, **kwargs):
    epsilon_factor = kwargs.get('epsilon_factor', 1)
    uniformity = kwargs.get('uniformity', True)
    complex_flag = False
    AffinityMat = AD.AffinityKernels(D, uniformity=uniformity, epsilon_factor=epsilon_factor)
    Embedding, _, eigenval_vec = AD.DiffusionMap(AffinityMat)
    complex_flag = np.any(np.iscomplex(Embedding))
    if complex_flag:
        logging.info("Diffusion Maps Decomposion: Results are Complex array")
        Embedding = np.real(Embedding)
        eigenval_vec = np.real(eigenval_vec)

    if kwargs.get('plot', False):
        plt.figure()
        plt.plot(eigenval_vec, 'ro')
        plt.title('Diffusion Maps Eigenvalues')

    return {'embedding': Embedding, 'eigenvals': eigenval_vec, 'complex': complex_flag}


def Spectral_Embedding(data, n_components=2):
    spectral_obj = SpectralEmbedding(n_components=n_components, affinity='precomputed')
    embedding = spectral_obj.fit_transform(data)
    return embedding


def PHATE(data, metric='precomputed', knn=5):
    phate_operator = phate.PHATE(mds_solver='smacof', knn_dist=metric, knn=knn)
    return phate_operator.fit_transform(data)


def PlotEmbedding(D, n_iter, perplexity):
    Embedding = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, metric='precomputed').fit_transform(D)
    return Embedding
