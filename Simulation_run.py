import numpy as np
import Geometric.AlternatingDiffusion as AD
import TDA.PersistentHomology as PH
from sklearn.manifold import MDS, TSNE
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from plotly import graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"


def GenerateTorusSamples(Type, N, angles, radius, noise=True):
    """
    generate samples on torus surface by Clifford Torus torus
    :param Type: which type of torus
    :param N: Dimension of the torus
    :param angles:
    :param radius:

    """
    NumSamples = angles.shape[1]

    if Type == 'Clifford':
        circle1D = lambda R, theta: np.array((R * np.cos(2 * np.pi * theta), R * np.sin(2 * np.pi * theta))).T
        TorusSamples = np.zeros((NumSamples, N * 2))
        for ii in range(N):
            TorusSamples[:, ii * 2: ii * 2 + 2] = circle1D(radius[ii], angles[ii, :])
    else:  # Cube
        TorusSamples = angles.T * np.tile(np.expand_dims(radius, axis=0), [NumSamples, 1])
    if noise:
        TorusSamples = TorusSamples + np.random.normal(0, 0.1, TorusSamples.shape)
    return TorusSamples


def DistanceMatTorus(Samples, radii):
    ###
    # This function calculates the distance between two points in the torus
    ###
    def distance_func(u, v, R=radii):
        return np.linalg.norm(np.min(np.vstack((np.abs(u - v), np.abs(u + R - v), np.abs(v + R - u))), axis=0))

    DIST = squareform(pdist(Samples, lambda u, v: distance_func(u, v)))
    return DIST


def ToriGraph(tori_dim, num_of_tori, angles_pool_size, base_radii, radius_STD, plot_graph=False):
    """
    ToriGraph will generate the relation between tori by setting the common angels
    :param ToriDim:
    :param NumOfTori:
    :return:
    """
    ToriAnglesRelations = np.random.randint(0, angles_pool_size, (num_of_tori, tori_dim), dtype=int)
    ToriRadiiSet = base_radii + np.random.normal(0, radius_STD, (num_of_tori, tori_dim))

    return ToriAnglesRelations, ToriRadiiSet


def ToriKernels(ToriAnglesRelations, RadiiSet, NumSamples, ToriiSamplesType, uniformity=False,
                std_normalization= False, affinity_threshold=None):
    NumAngels = np.max(ToriAnglesRelations) + 1  # add 1 because starting count angels from 0
    NumOfTori = ToriAnglesRelations.shape[0]
    TorusDimension = ToriAnglesRelations.shape[1]
    matSamplesAngle = np.random.uniform(0, 1, (NumAngels, TorusDimension, NumSamples))

    ToriKernels = [None] * NumOfTori

    for ii, torus in enumerate(ToriAnglesRelations):
        angels_samples = np.array([matSamplesAngle[torus[ind], ind, :] for ind in range(TorusDimension)])
        ToriSamples = GenerateTorusSamples(ToriiSamplesType, TorusDimension,
                                           angels_samples, RadiiSet[ii, :])

        #  uniformity factor (epsilon) for the affinity matrix as function of the std of samples
        if std_normalization:
            ToriSamples -= np.mean(ToriSamples, axis=0)
            epsilon_factor = np.std(ToriSamples, axis=0)
            ToriSamples = ToriSamples / epsilon_factor

        if ToriiSamplesType == 'Clifford':
            ToriKernels[ii] = AD.GetDiffusionMapsKernel(ToriSamples,
                                                        epsilon_factor=1,
                                                        metric='euclidean',
                                                        uniformity=uniformity,
                                                        affinity_threshold=affinity_threshold)
        else:  # Cube troii
            ToriKernels[ii] = AD.GetDiffusionMapsKernel(ToriSamples,
                                                        epsilon_factor=1,
                                                        metric='custom',
                                                        custom_func=DistanceMatTorus,
                                                        radii_set=RadiiSet[ii, :],
                                                        uniformity=uniformity)
    return ToriKernels

def Main():
    # np.random.seed(42)
    ConnectivityAnalysis = True
    MultiDiag = True

    ToriiSamplesType = 'Clifford'  # Clifford Cube
    Norm = 'fro'  # nuc fro
    Diffusion_dist = False  # True False
    WeightFunc = 'Inverse'  # Inverse  NormalizeInverse  LogInverse  Identity
    # kernel multiplication order
    WeightingMethod = 'addition'  # direct  symmetric  eigenvalues addition subtraction
    SamplesSTDNormalization = True  # False
    # uniformity affinity operator for uniform distribution
    NormalizeDiffOp = True
    DiagramDistance = 'wasserstein'  # bottleneck wasserstein
    AffinitiyTreshold = 1e-5

    PH_dimension = 2
    NumOfTori = 30
    TorusDimension = 3
    NumSamples = 200

    BaseRadii = (15, 7, 1)
    RadiusSTD = 0.01
    RadiusSTD_vec = np.array([0.001, 0.01, 0.1, 1, 10, 100, 1000])

    # multiple simplexes parameters
    AnglesPoolSize = 30
    NumOfRadiiIterations = 5

    if ConnectivityAnalysis:
        SamplesSet = ['Clifford']
        NormSet = ['nuc', 'fro']
        DiffDistSet = [False, True]
        WeightingSet = ['addition']  # 'eigenvalues'
        WeightFuncSet = ['Inverse', 'NormalizeInverse', 'LogInverse', 'Identity']
        NormalizeDiffOpSet = [True, False]

        radius_range = [1, 15]
        Num_radius_iter = 30
        radii_set = np.vstack((BaseRadii[0] * np.ones((Num_radius_iter)),
                                  np.linspace(BaseRadii[1], BaseRadii[0], Num_radius_iter),
                                  np.linspace(BaseRadii[2], BaseRadii[0], Num_radius_iter))).T

        ToriAnglesRelations = [np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
                               np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]]),
                               np.array([[0, 0, 0], [0, 1, 1], [0, 2, 2]]),
                               np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
                               np.array([[0, 0, 0], [0, 1, 1], [1, 1, 0]])]

        KernelOrder = [[0, 1, 2],
                       [2, 0, 1],
                       [1, 2, 0]]

        edges_radius = np.zeros((4, 2))
        faces_radius = np.zeros((4, 2))
        n_rep = 20
        WeightingType = WeightingSet[0]
        SamplesType = SamplesSet[0]
        NormalizeDiffOptType = NormalizeDiffOpSet[0]
        NormType = NormSet[1]
        WeightFuncType = WeightFuncSet[0]
        DiffDistType = DiffDistSet[0]
        WeightHandle = PH.WeightFuncSelection(WeightFuncType)

        # R_max experiment
        for cc, ToriAngles in enumerate(ToriAnglesRelations):
            edges_weights = np.zeros((Num_radius_iter, len(KernelOrder)))
            face_weights  = np.zeros(Num_radius_iter)

            for _ in range(n_rep):
                for ii, radius in enumerate(radii_set):
                    Tori_radii = np.ones((3, 1)) * radius + np.random.normal(0, RadiusSTD, (3, 1))
                    ToriDiffKernels = ToriKernels(ToriAngles, Tori_radii, NumSamples,
                                                  SamplesType,
                                                  uniformity=NormalizeDiffOptType,
                                                  std_normalization=SamplesSTDNormalization)
                    Face_AD_Op = np.zeros((NumSamples, NumSamples))
                    for jj, kernel_iter in enumerate(KernelOrder):
                        Edge_AD_Op = PH.KernelMultiplication([ToriDiffKernels[kernel_iter[0]],
                                                              ToriDiffKernels[kernel_iter[1]]],
                                                             Method=WeightingType)
                        Face_AD_Op += PH.KernelMultiplication([Edge_AD_Op,
                                                               ToriDiffKernels[kernel_iter[2]]],
                                                              Method=WeightingType)
                        Edge_Op_norm = PH.MatrixNorm(Edge_AD_Op, NormType, diff_dist=DiffDistType)
                        edges_weights[ii, jj] += WeightHandle(Edge_Op_norm)


                    Face_AD_Op = Face_AD_Op / 3  # uniformity the sum of 3 kernels
                    Face_Op_norm = PH.MatrixNorm(Face_AD_Op, NormType, diff_dist=DiffDistType)
                    face_weights[ii] += WeightHandle(Face_Op_norm)

            if not cc == 4:
                edges_radius[cc, 0] = np.mean(np.log10(1 - (edges_weights[:, 0] / n_rep)))
                edges_radius[cc, 1] = np.std(np.log10(1 - (edges_weights[:, 0] / n_rep)))
                faces_radius[cc, 0] = np.mean(np.log10(1 - (face_weights[face_weights < n_rep] / n_rep)))
                faces_radius[cc, 1] = np.std(np.log10(1 - (face_weights[face_weights < n_rep] / n_rep)))


        n_noise = RadiusSTD_vec.shape[0]
        edges_std = np.zeros((4, 2))
        faces_std = np.zeros((4, 2))
        # sigma_i experiment
        for cc, ToriAngles in enumerate(ToriAnglesRelations):
            edges_weights = np.zeros((n_noise, len(KernelOrder)))
            face_weights = np.zeros(n_noise)
            WeightHandle = PH.WeightFuncSelection(WeightFuncType)

            for _ in range(n_rep):
                radius = np.array([1, 1, 1])
                for ii, RadiusSTD in enumerate(RadiusSTD_vec):
                    Tori_radii = np.ones((3, 1)) * radius + np.random.normal(0, RadiusSTD, (3, 1))
                    ToriDiffKernels = ToriKernels(ToriAngles, Tori_radii, NumSamples,
                                                  SamplesType,
                                                  uniformity=NormalizeDiffOptType,
                                                  std_normalization=SamplesSTDNormalization)
                    Face_AD_Op = np.zeros((NumSamples, NumSamples))
                    for jj, kernel_iter in enumerate(KernelOrder):
                        Edge_AD_Op = PH.KernelMultiplication([ToriDiffKernels[kernel_iter[0]],
                                                              ToriDiffKernels[kernel_iter[1]]],
                                                             Method=WeightingType)
                        Face_AD_Op += PH.KernelMultiplication([Edge_AD_Op,
                                                               ToriDiffKernels[kernel_iter[2]]],
                                                              Method=WeightingType)
                        Edge_Op_norm = PH.MatrixNorm(Edge_AD_Op, NormType, diff_dist=DiffDistType)
                        edges_weights[ii, jj] += WeightHandle(Edge_Op_norm)


                    Face_AD_Op = Face_AD_Op / 3  # uniformity the sum of 3 kernels
                    Face_Op_norm = PH.MatrixNorm(Face_AD_Op, NormType, diff_dist=DiffDistType)
                    face_weights[ii] += WeightHandle(Face_Op_norm)

            if not cc == 4:
                edges_std[cc, 0] = np.mean(np.log10(1 - (edges_weights[:, 0] / n_rep)))
                edges_std[cc, 1] = np.std(np.log10(1 - (edges_weights[:, 0] / n_rep)))
                faces_std[cc, 0] = np.mean(np.log10(1 - (face_weights[face_weights < n_rep] / n_rep)))
                faces_std[cc, 1] = np.std(np.log10(1 - (face_weights[face_weights < n_rep] / n_rep)))


        fig_bar = go.Figure(data=[
            go.Bar(name=r"$V([i_1,i_2])_{(\sigma_i)}$",
                   x=np.array([3, 2, 1, 0]),
                   y=edges_std[:, 0],
                   error_y=dict(type='data', array=edges_std[:, 1]),
                   textposition='auto'),
            go.Bar(name=r"$V([i_1,i_2])_{(R_{\text{max}})}$",
                   x=np.array([3, 2, 1, 0]),
                   y=edges_radius[:, 0],
                   error_y=dict(type='data', array=edges_radius[:, 1]),
                   textposition='auto'),
            go.Bar(name=r"$V([i_1,i_2,i_3])_{(\sigma_i)}$",
                   x=np.array([3, 2, 1, 0]),
                   y=faces_std[:, 0],
                   error_y=dict(type='data', array=faces_std[:, 1]),
                   textposition='auto'),
            go.Bar(name=r"$V([i_1,i_2,i_3])_{(R_{\text{max}})}$",
                   x=np.array([3, 2, 1, 0]),
                   y=faces_radius[:, 0],
                   error_y=dict(type='data', array=faces_radius[:, 1]),
                   textposition='auto'),

        ])
        # Change the bar mode
        fig_bar.update_layout(barmode='group',
                              xaxis_title=r"$\text{# Common Manifolds}$",
                              yaxis_title=r"[dB]",
                              font=dict(size=20,
                                        color='black'),
                              width=650,
                              height=450,
                              xaxis=dict(
                                  tickmode='array',
                                  tickvals=[0, 1, 2, 3],
                                  ticktext=['None', 'One', 'Two', 'Three'])
                              )
        fig_bar.update_yaxes(autorange="reversed")
        fig_bar.show()

    # Analysis of a full algorithm from multiple simplexes
    if MultiDiag:
        DiagramList = [None] * (AnglesPoolSize * NumOfRadiiIterations)
        matDiagramDistance = [None] * PH_dimension
        # preparing sets of tori base radii and angle pool size
        vecAnglesPoolSize = np.arange(2, AnglesPoolSize + 2)
        AnglesPoolSizeSet = np.tile(vecAnglesPoolSize, NumOfRadiiIterations)
        vecBaseRadii = np.vstack((BaseRadii[0] * np.ones((NumOfRadiiIterations)),
                                  np.linspace(BaseRadii[1], BaseRadii[0], NumOfRadiiIterations),
                                  np.linspace(BaseRadii[2], BaseRadii[0], NumOfRadiiIterations))).T
        BaseRadiiSet = vecBaseRadii.repeat(AnglesPoolSize, axis=0)

        # generate persistent diagram list from all graph from growing angle pool size and base radii
        for ii, base_radii in enumerate(tqdm(BaseRadiiSet)):
            # connection graph between angles in order to set tori angles relations
            ToriAnglesRelations, ToriRadiiSet = ToriGraph(TorusDimension,
                                                          NumOfTori,
                                                          AnglesPoolSizeSet[ii],
                                                          base_radii,
                                                          RadiusSTD,
                                                          plot_graph=False)
            # generate tori samples with the connection above setting the the angles of each tori
            ToriDiffKernels = ToriKernels(ToriAnglesRelations, ToriRadiiSet,
                                          NumSamples, ToriiSamplesType,
                                          uniformity=NormalizeDiffOp,
                                          std_normalization=SamplesSTDNormalization,
                                          affinity_threshold=AffinitiyTreshold)
            # build simplex from tori and get persistence diagram
            DiagramList[ii] = PH.PersistentDiagram(ToriDiffKernels,
                                                   np.arange(len(ToriDiffKernels)),
                                                   KernelNorm=Norm,
                                                   WeightFunc=WeightFunc,
                                                   WeighingMethod=WeightingMethod,
                                                   Diff_Dist=Diffusion_dist)


        # generate persistent diagram distance matrix use bottleneck distance
        for dim in range(PH_dimension):
            matDiagramDistance[dim] = PH.DiagramDistance(DiagramList, dim, distance=DiagramDistance)

        embedding = TSNE(n_components=2,
                                 perplexity=20,
                                 n_iter=3000,
                                 metric='precomputed',
                                 ).fit_transform(matDiagramDistance[1])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=embedding[:, 0],
                                 y=embedding[:, 1],
                                 mode='markers',
                                 showlegend=False,
                                 marker=dict(size=8,
                                             color=AnglesPoolSizeSet,
                                             colorbar=dict(thickness=18))))
        fig.update_layout(font=dict(size=20,
                                    color='black'),
                          width=700,
                          height=550,
                          xaxis_title="tSNE Axis 1",
                          yaxis_title="tSNE Axis 2",)
        fig.show()


if __name__ == '__main__':
    Main()
