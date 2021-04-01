import numpy as np
import logging
import pandas as pd
import os
import glob
import Geometric.AlternatingDiffusion as AD
import TDA.PersistentHomology as PH
import Geometric.DimensionReduction as DR
from sklearn.manifold import TSNE, MDS
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.distance import cdist, pdist, squareform
from tqdm import tqdm
from HSI_utils.Classifiers import Classification
from Utils.utils import im2col, get_vertex_neighbors, split_image, get_valid_patches
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import spectral
import spectral.io.envi as envi


class HSI_Data:
    data_dir = os.getcwd() + '/ICONES_Data/ICONES-HSI/'
    data_pickle_dir = os.getcwd() + '/ICONES_Data/ICONES-HSI_pickle/'

    labels_names = ['Agriculture', 'Cloud', 'Desert', 'Dense_urban',
                    'Forest', 'Mountain', 'Ocean', 'Snow', 'Wetland']

    labels   = []
    ID       = []
    DiagList = []
    data_counter = 0

    PH_dimension = 2
    matDist = [None] * PH_dimension
    DM_embedding = [None] * PH_dimension
    MDS_embedding = {}

    def __init__(self, conf, patch_size, eps_factor, graph_type,
                 generate_diag=True,
                 geometry_run=False,
                 homology_run=False):
        self.conf = conf
        self.kernel_distance           = conf['kernel_distance']
        self.weighting_method          = conf['weighting_method']
        self.operator_norm             = conf['operator_norm']
        self.weight_func               = conf['weight_func']
        self.uniform_affinity_operator = conf['uniform_affinity_operator']
        self.save_figure               = conf['save_figures']
        self.run_result_dir            = conf['run_result_dir']
        self.save_diagram_list         = conf['save_embedding']
        self.sample_process            = conf['sample_process']
        self.low_of_three              = conf['low_of_three']
        self.diagram_type              = conf['diagram_type']
        self.ignore_subjects           = np.array(conf['ignore_subjects'])
        self.save_diagram_dist         = conf['save_embedding']
        self.patch_size                = patch_size
        self.eps_factor                = eps_factor
        self.graph_type                = graph_type

        # load and preprocess data
        logging.info('Initialize data Salinas in object: START')
        logging.info('Start Attribute Presistent Homology')
        logging.info('Diagram type: %s' % self.diagram_type)
        logging.info('Graph type: %s' % self.graph_type)
        self.scaler = StandardScaler()
        if conf['init_data']:
            # clean old diagram list
            if len(self.DiagList):
                self.DiagList = []

            # read data and labels
            with tqdm(total=486) as pbar:
                for label_idx, label_dir in enumerate(self.labels_names):
                    img_files = glob.glob('%s/%s/*.img' % (self.data_dir, label_dir))
                    hdr_files = glob.glob('%s/%s/*.hdr' % (self.data_dir, label_dir))
                    for file in zip(hdr_files, img_files):
                        self.labels.append(label_idx)
                        img_header = envi.open(file[0], file[1])
                        img = np.array(img_header.open_memmap(writeable=True))
                        if not img.shape == (300, 300, 224):
                            logging.info('Bad File: %s. Image shape %s' % (file[1], img.shape))
                            new_path = file[1].replace(self.data_dir, self.data_pickle_dir)
                            new_path = new_path.replace('.img', '.pickle')
                            with open(new_path, 'rb') as f:
                                img = pickle.load(f)
                        self.data_counter += 1
                        if generate_diag:
                            self.AttributePersistentDiagram(img)
                        elif geometry_run:
                            self.Geometry(img)
                        elif homology_run:
                            self.Homology(img)
                        pbar.update(1)

            self.n_channels = img.shape[-1]

        self.n_attribute = self.data_counter
        self.attribute_vec = np.arange(self.n_attribute)

        logging.info('Initialize data in object: FINISH')

    def GenerateKernels(self, data):
        """
        GenerateKernels calculate list of kernels for given attribute. kernel is from all sensors for the specific
        attribute
        :param Attribute:
        :return:
        """

        n_channels = data.shape[2]
        image_size = data.shape[:2]
        data = np.expand_dims(data, axis=0).transpose([3, 0, 1, 2])
        pad_needed = np.mod(self.patch_size - np.mod(image_size, self.patch_size), self.patch_size)
        node_neighbors_vec = None

        patches, grid_size = im2col(data,
                                    self.patch_size,
                                    self.patch_size,
                                    stride=int(self.patch_size * self.conf['stride']),
                                    pad=pad_needed)
        n_patches = np.prod(grid_size)

        valid_idx = np.arange(patches.shape[1])

        if self.graph_type == 'spatial':

            KernelList = [None] * n_patches
            node_neighbors_vec = get_vertex_neighbors(grid_size, valid_idx)

            # to here neighbor
            for patch_idx in valid_idx:
                KernelList[patch_idx] = AD.SingleSensorDiffusionKernel(patches[:, patch_idx, :],
                                                                       sample_preprocess=self.sample_process,
                                                                       kernel_distance=self.kernel_distance,
                                                                       uniformity=self.uniform_affinity_operator,
                                                                       patch_size=self.patch_size,
                                                                       epsilon_factor=self.eps_factor,
                                                                       normalize_scale=True,
                                                                       plot=False)
        elif self.graph_type == 'spectral':
            KernelList = [None] * n_channels
            for spectral_idx in range(n_channels):
                KernelList[spectral_idx] = AD.SingleSensorDiffusionKernel(patches[spectral_idx, :, :],
                                                                          sample_preprocess=self.sample_process,
                                                                          kernel_distance=self.kernel_distance,
                                                                          uniformity=self.uniform_affinity_operator,
                                                                          patch_size=self.patch_size,
                                                                          epsilon_factor=self.eps_factor,
                                                                          plot=False)


        return KernelList, grid_size, node_neighbors_vec, valid_idx

    def AttributePersistentDiagram(self, data):
        """

        :return:
        """
        AttributeKernels, grid_size, node_neighbors_vec, valid_idx_vec = self.GenerateKernels(data)
        # location of the kernels
        kernels_location = np.array(np.unravel_index(np.arange(np.prod(grid_size)), grid_size)).T

        if self.diagram_type == 'diffusion_kernel_fully':
            self.DiagList.append(PH.PersistentDiagram(AttributeKernels,
                                                      valid_idx_vec,
                                                      KernelNorm=self.operator_norm,
                                                      WeighingMethod=self.weighting_method,
                                                      WeightFunc=self.weight_func,
                                                      low_of_three=self.low_of_three,
                                                      kernel_location=kernels_location,
                                                      Plot=False))

        elif self.diagram_type == 'diffusion_kernel_neighbors':
            self.DiagList.append(PH.PersistentNeighborsDiagram(AttributeKernels,
                                                               valid_idx_vec,
                                                               node_neighbors_vec,
                                                               KernelNorm=self.operator_norm,
                                                               WeighingMethod=self.weighting_method,
                                                               WeightFunc=self.weight_func,
                                                               low_of_three=self.low_of_three,
                                                               kernel_location=kernels_location,
                                                               Plot=False))

        if self.diagram_type == 'click_complex':
            for ii, attr in tqdm(enumerate(self.attribute_vec)):
                electrodes_data = self.data.loc[attr]
                electrodes_data = electrodes_data.loc['Data'].T
                self.DiagList[ii] = PH.BuildClickComplex(electrodes_data,
                                                         KernelNorm=self.operator_norm,
                                                         WeighingMethod=self.weighting_method,
                                                         WeightFunc=self.weight_func,
                                                         Plot=False)

    def GenerateDiagDistance(self,
                             diagram_distance,
                             matrix_counter,
                             clean_small_homology=None):

        for dim in range(self.PH_dimension):
            logging.info("H_%d distance matrix" % dim)
            self.matDist[dim] = PH.DiagramDistance(self.DiagList,
                                                   dim,
                                                   distance=diagram_distance,
                                                   clean_small_homology=clean_small_homology)

        logging.info("Plotting %s distance matrix" % diagram_distance)
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("$\\beta_0$", "$\\beta_1$"))
        fig.add_trace(go.Heatmap(z=self.matDist[0],
                                 colorbar=dict(x=0.45),
                                 colorscale="Viridis"),
                      row=1, col=1)
        fig.add_trace(go.Heatmap(z=self.matDist[1],
                                 colorscale="Viridis"),
                      row=1, col=2)
        fig.update_layout(yaxis_autorange='reversed',
                          yaxis2_autorange='reversed',
                          title_text=diagram_distance)
        fig.write_html("%s/%d.html" % (self.conf['matrix_distance_dir'], matrix_counter))
        return self.matDist

    def DiffusionMapsDiagProjection(self, epsilon_factor=1):

        self.DM_embedding = [None] * self.PH_dimension

        for dim in range(self.PH_dimension):
            self.DM_embedding[dim] = DR.DiffusionMaps(self.matDist[dim], epsilon_factor=epsilon_factor)

    def MDS_DiagProjection(self):
        MetricMDS = MDS(n_components=2, max_iter=1000, eps=1e-6,
                            dissimilarity="precomputed", n_jobs=4)
        self.MDS_embedding['Metric']  = [MetricMDS.fit(self.matDist[0]).embedding_,
                                         MetricMDS.fit(self.matDist[1]).embedding_]
        NonMetricMDS = MDS(n_components=2, metric=False, max_iter=1000, eps=1e-6,
                           dissimilarity="precomputed", n_jobs=4, n_init=1)
        self.MDS_embedding['Non-Metric'] = [NonMetricMDS.fit_transform(self.matDist[0],
                                                                       init=self.MDS_embedding['Metric'][0]),
                                            NonMetricMDS.fit_transform(self.matDist[1],
                                                                       init=self.MDS_embedding['Metric'][1])]

    def PlotEmbedding(self,
                      patch_size,
                      diagram_distance,
                      **kwargs):

        DM_plot = kwargs.get('DM_plot', False)
        MDS_plot = kwargs.get('MDS_plot', False)
        tSNE_plot = kwargs.get('tSNE_plot', False)
        PHATE_plot = kwargs.get('PHATE_plot', False)
        SpectralEmbedding_plot = kwargs.get('Spectral_plot', False)
        additional_title = kwargs.get('title')
        DM_vec_idx = np.arange(0, 6)
        perplexity = [2, 5, 10, 20]
        n_iter = 2000
        knn_vec = [3, 5, 8]
        DM_eps_factor = [0.5, 1, 2]

        if tSNE_plot:
            TSNE_embedding = [None] * self.PH_dimension * len(perplexity)
            for p_idx, p in enumerate(perplexity):
                TSNE_embedding[p_idx * self.PH_dimension] = TSNE(n_components=2,
                                                                 perplexity=p,
                                                                 n_iter=n_iter,
                                                                 metric='precomputed').fit_transform(self.matDist[0])
                TSNE_embedding[p_idx * self.PH_dimension + 1] = TSNE(n_components=2,
                                                                     perplexity=p,
                                                                     n_iter=n_iter,
                                                                     metric='precomputed').fit_transform(self.matDist[1])
            self.auxPlot(TSNE_embedding,
                         Title=r't-SNE %d %s %s' % (patch_size, diagram_distance, additional_title),
                         figTitle="t-SNE_{}_{}".format(patch_size, diagram_distance),
                         hyper_param=perplexity,
                         hyper_params_name='p',
                         plot_type='tSNE')

        if PHATE_plot:
            PHATE_embedding = [None] * self.PH_dimension * len(knn_vec)
            for knn_ind, knn in enumerate(knn_vec):
                PHATE_embedding[knn_ind * self.PH_dimension] = DR.PHATE(self.matDist[0], knn=knn)
                PHATE_embedding[knn_ind * self.PH_dimension + 1] = DR.PHATE(self.matDist[1], knn=knn)
            self.auxPlot(PHATE_embedding,
                         Title=r'PHATE ' + str(patch_size) +
                               " {} {}".format(diagram_distance, additional_title),
                         figTitle="PHATE_{}_{}_{}".format(patch_size, knn_vec, diagram_distance),
                         hyper_param=knn_vec,
                         hyper_params_name='knn',
                         plot_type='PHATE')


        if DM_plot:
            for eps_factor in DM_eps_factor:
                self.DiffusionMapsDiagProjection(epsilon_factor=eps_factor)
                self.auxPlot(self.DM_embedding, Title='Diffusion Maps e = %.2f - %d %s %s'
                                                      % (eps_factor, patch_size, diagram_distance, additional_title),
                             figTitle="DM {} {} eps {}".format(patch_size, diagram_distance, eps_factor),
                             plot_type='DM')
                DM_TSNE = [None] * self.PH_dimension * len(perplexity)

                # DM t-SNE
                for p_idx, p in enumerate(perplexity):
                    DM_TSNE[p_idx * self.PH_dimension] = TSNE(n_components=2,
                                                              perplexity=p,
                                                              n_iter=n_iter).fit_transform(
                        self.DM_embedding[0]['embedding'][:, DM_vec_idx])
                    DM_TSNE[p_idx * self.PH_dimension + 1] = TSNE(n_components=2,
                                                                  perplexity=p,
                                                                  n_iter=n_iter).fit_transform(
                        self.DM_embedding[1]['embedding'][:, DM_vec_idx])

                self.auxPlot(DM_TSNE,
                             Title='Diffusion Maps e = %.2f (%d vectors) t-SNE %d %s %s'
                                   % (eps_factor, DM_vec_idx.shape[0], patch_size, diagram_distance, additional_title),
                             figTitle="DM - tSNE {} {} eps {}".format(patch_size, diagram_distance, eps_factor),
                             hyper_param=perplexity,
                             hyper_params_name='p',
                             plot_type='tSNE')

                # DM PHATE
                if PHATE_plot:
                    try:
                        for knn_ind, knn in enumerate(knn_vec):
                            PHATE_embedding[knn_ind * self.PH_dimension] = DR.PHATE(
                                self.DM_embedding[0]['embedding'][:, DM_vec_idx],
                                metric='euclidean',
                                knn=knn)
                            PHATE_embedding[knn_ind * self.PH_dimension + 1] = DR.PHATE(
                                self.DM_embedding[1]['embedding'][:, DM_vec_idx],
                                metric='euclidean',
                                knn=knn)
                        self.auxPlot(PHATE_embedding,
                                     Title=r'Diffusion Maps e = %f (5 vectors) PHATE ' %  eps_factor
                                           + str(patch_size) + " {} {}".format(diagram_distance, additional_title),
                                     figTitle="DM - PHATE_{}_{}_{}_eps {}".format(patch_size, p, diagram_distance, eps_factor),
                                     hyper_param=knn_vec,
                                     hyper_params_name='knn',
                                     plot_type = 'PHATE')
                    except:
                        logging.info('Failure in DM PHATE')

        if MDS_plot:
            self.MDS_DiagProjection()
            self.auxPlot(self.MDS_embedding,
                         Title='MDS  -  %d %s %s' % (patch_size, diagram_distance, additional_title),
                         figTitle="MDS {} {}".format(patch_size, diagram_distance),
                         plot_type='MDS')

        if SpectralEmbedding_plot:
            self.auxPlot([DR.Spectral_Embedding(self.matDist[0]),
                          DR.Spectral_Embedding(self.matDist[1])],
                         Title=r'Spectral embedding - %d %s %s' % (patch_size, diagram_distance, additional_title),
                         figTitle="SE {} {}".format(patch_size, diagram_distance),
                         plot_type='SpectralEmbedding')


    def auxPlot(self, matEmbeddingList, **kwargs):

        figTitle          = kwargs.get('figTitle', None)
        Title             = kwargs.get('Title', None)
        hyper_param       = kwargs.get('hyper_param', None)
        hyper_params_name = kwargs.get('hyper_params_name', None)
        plot_type         = kwargs.get('plot_type', None)

        colormap = px.colors.qualitative.Dark24

        # tSNE ploting and PHATE:
        if plot_type in ['tSNE', 'PHATE']:

            fig = make_subplots(rows=self.PH_dimension, cols=len(hyper_param),
                                subplot_titles=["%s = %d" %(hyper_params_name, val) for val in hyper_param] * self.PH_dimension,
                                vertical_spacing=0.04,
                                horizontal_spacing=0.06)
            for embedd_idx, matEmbedding in enumerate(matEmbeddingList):
                # plot Responder label
                fig.add_trace(go.Scatter(x=matEmbedding[:, 0], y=matEmbedding[:, 1],
                                         mode='markers',
                                         marker=dict(color=self.labels,
                                                     size=12,
                                                     colorscale=colormap),
                                         showlegend=False,
                                         text=self.labels),
                              row=embedd_idx % self.PH_dimension + 1,
                              col=embedd_idx // self.PH_dimension + 1)

        # plot Diffusion Maps:
        elif plot_type == 'DM':
            fig = make_subplots(rows=self.PH_dimension, cols=2,
                                subplot_titles=['Embedding', 'Eigenvalues'],
                                vertical_spacing=0.04,
                                horizontal_spacing=0.06)
            for embedd_idx, matEmbeddingDict in enumerate(matEmbeddingList):
                fig.add_trace(go.Scatter(x=matEmbeddingDict['embedding'][:, 0], y=matEmbeddingDict['embedding'][:, 1],
                                         mode='markers',
                                         marker=dict(color=self.labels,
                                                     size=12,
                                                     colorscale=colormap),
                                         showlegend=False,
                                         text=self.labels),
                              row=embedd_idx + 1,
                              col=1)
                fig.add_trace(go.Scatter(x=np.arange(matEmbeddingDict['eigenvals'].shape[0]),
                                         y=matEmbeddingDict['eigenvals'],
                                         mode='markers',
                                         marker=dict(size=12),
                                         showlegend=False),
                              row=embedd_idx + 1,
                              col=2)

        # plot MDS
        elif plot_type == 'MDS':
            fig = make_subplots(rows=self.PH_dimension, cols=2,
                                subplot_titles=["\\text{Metric} $\\beta_0$", "\\text{Non-Metric} $\\beta_0$",
                                                "\\text{Metric} $\\beta_1$", "\\text{Metric} $\\beta_1$"],
                                vertical_spacing=0.04,
                                horizontal_spacing=0.06)
            ii = 1
            for key, dim_embedding in matEmbeddingList.items():
                for embedd_idx, matEmbedding in enumerate(dim_embedding):
                    fig.add_trace(go.Scatter(x=matEmbedding[:, 0], y=matEmbedding[:, 1],
                                             mode='markers',
                                             marker=dict(color=self.labels,
                                                         size=12,
                                                         colorscale=colormap),
                                             showlegend=False,
                                             text=self.labels),
                                  row=embedd_idx + 1,
                                  col=ii)
                ii += 1

        elif plot_type == 'SpectralEmbedding':
            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=["$\\beta_0$", "$\\beta_1$"],
                                vertical_spacing=0.04,
                                horizontal_spacing=0.06)
            for embedd_idx, matEmbedding in enumerate(matEmbeddingList):
                fig.add_trace(go.Scatter(x=matEmbedding[:, 0], y=matEmbedding[:, 1],
                                         mode='markers',
                                         marker=dict(color=self.labels,
                                                     size=12,
                                                     colorscale=colormap),
                                         showlegend=False,
                                         text=self.labels),
                              row=1,
                              col=embedd_idx + 1)


        fig.update_layout(title_text=Title,
                          yaxis_title="$\\beta$")
        fig.show()
        if self.save_figure:
            fig.write_html("%s/%s.html" % (self.run_result_dir, figTitle))

    def Classification(self, classifier='SVM'):
        ClassificationObj = Classification()
        DM_features = [2, 3, 4, 5, 8, 10, 15]
        DM_eps_factor = [0.1, 0.5, 1, 2, 5]
        best_score = -1
        best_params = None
        best_eps = None
        best_n = None
        params = None
        for eps_factor in DM_eps_factor:
            self.DiffusionMapsDiagProjection(epsilon_factor=eps_factor)
            for ii in range(self.PH_dimension):
                embedding = self.DM_embedding[ii]['embedding']
                for n_features in DM_features:
                    try:
                        score, params = ClassificationObj.SVM(embedding[:, 0:n_features], self.labels)
                    except:
                        score = 0
                        params = None
                    if best_score < score:
                        best_score = score
                        best_params = params
                        best_eps = eps_factor
                        best_dim = ii
                        best_n = n_features
        logging.info("DM classification: score: %.3f" % best_score)
        logging.info("H dim: %d  |  eps: %.3f  |  n features: %d  |  SVM: %s" % (best_dim, best_eps, best_n, best_params))

        return best_score

    def __getitem__(self, *args):
        if args[0] == 'diffusion_maps':
            return self.DM_embedding
        if args[0] == 'MDS':
            return self.MDS_embedding
        if args[0] == 'responders':
            return self.labels
        if args[0] == 'data':
            return self.data

    def Geometry(self, data):

        Kernels, grid_size, node_neighbors_vec, valid_idx_vec = self.GenerateKernels(data)
        kernels_location = np.array(np.unravel_index(np.arange(np.prod(grid_size)), grid_size)).T

        weights_edges_dict = {}
        WeightHandle = lambda x : 1 / x

        n_kernels = valid_idx_vec.shape[0]

        # calculate and insert edges weights to simplex
        edge_weights_mat = np.zeros((n_kernels, n_kernels))
        for node_idx, neighbors in zip(valid_idx_vec, node_neighbors_vec):
            neighbors = neighbors['edges']
            for edge_neighbor_idx in neighbors:
                two_neighbors = np.sort([node_idx, edge_neighbor_idx])
                two_neighbors = (two_neighbors[0], two_neighbors[1])
                if two_neighbors not in weights_edges_dict.keys():
                    EdgeOperator = PH.KernelMultiplication([Kernels[two_neighbors[0]],
                                                            Kernels[two_neighbors[1]]],
                                                           Method=self.weighting_method)
                    EdgeOperatorNorm = PH.MatrixNorm(EdgeOperator, self.operator_norm, diff_dist=False)
                    weight = WeightHandle(EdgeOperatorNorm)
                    edge_weights_mat[two_neighbors[0], two_neighbors[1]] = weight
                    edge_weights_mat[two_neighbors[1], two_neighbors[0]] = weight
                    weights_edges_dict[two_neighbors] = weight

        self.DiagList.append(edge_weights_mat)

    def GeometryDistanceMatrix(self):
        # compute eigenvalues for all matrices
        eigenvalues = np.zeros((self.n_attribute, self.DiagList[0].shape[0]))
        for idx in range(self.n_attribute):
            eigenvalues[idx, :], _ = np.linalg.eig(self.DiagList[idx])
        dist_mat = squareform(pdist(eigenvalues))

        return [dist_mat]


    def Homology(self, data):
        """
        Phase 2 of the Disassemble inquiry construct diffusion kernels for each sensor in each attribute and
        pairwise distance between all attributes
        :return:
        """
        image_size = data.shape[:2]
        n_channels = data.shape[2]
        # scale data
        data = data.reshape(-1, data.shape[-1])
        data = self.scaler.fit_transform(data).reshape(image_size[0], image_size[1], n_channels)
        # divide to patches
        data = np.expand_dims(data, axis=0).transpose([3, 0, 1, 2])
        pad_needed = np.mod(self.patch_size - np.mod(image_size, self.patch_size), self.patch_size)

        patches, grid_size = im2col(data,
                                    self.patch_size,
                                    self.patch_size,
                                    stride=int(self.patch_size * self.conf['stride']),
                                    pad=pad_needed)
        n_patches = np.prod(grid_size)
        valid_idx = np.arange(patches.shape[1])
        node_neighbors_vec = get_vertex_neighbors(grid_size, valid_idx)
        kernels_location = np.array(np.unravel_index(np.arange(n_patches), grid_size)).T

        patches = patches.transpose([1, 0, 2])
        patches = np.reshape(patches, (patches.shape[0], -1))

        self.DiagList.append(PH.CovarianceNeighborsClickComplex(patches,
                                                                valid_idx,
                                                                node_neighbors_vec,
                                                                KernelNorm=self.operator_norm,
                                                                WeighingMethod=self.weighting_method,
                                                                WeightFunc=self.weight_func,
                                                                low_of_three=self.low_of_three,
                                                                kernel_location=kernels_location,
                                                                Plot=False))

