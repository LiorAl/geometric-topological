import pickle
import numpy as np
import glob
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import DimensionReduction as DR
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.metrics import auc, accuracy_score, confusion_matrix, average_precision_score, make_scorer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.manifold import TSNE
import umap
import warnings

pio.renderers.default = "browser"
labels_names = ['Agriculture', 'Cloud', 'Desert', 'Dense-Urban',
                'Forest', 'Mountain', 'Ocean', 'Snow', 'Wetland']

def missing_param_run_function():
    # with open('ICONES_Data/scores.pickle', 'rb') as fileH:
    #     scores_df = pickle.load(fileH)

    scores_df = pd.read_json('ICONES_Data/scores.json')

    wasserstein = scores_df['distance'].values == 'wasserstein'
    bottleneck = scores_df['distance'].values == 'bottleneck'

    def check_func(uniform_affinity, patch, eps):
        uniform_valid_idx = scores_df['uniform_affinity_operator'].values == uniform_affinity
        patch_size_valid_idx = scores_df['patch_size'].values == patch
        epsilon_factor_valid_idx = scores_df['epsilon_factor'].values == eps
        wasserstein_valid_idx = uniform_valid_idx & wasserstein & patch_size_valid_idx & epsilon_factor_valid_idx
        bottleneck_valid_idx = uniform_valid_idx & bottleneck & patch_size_valid_idx & epsilon_factor_valid_idx
        distance_vec = []
        if not np.any(wasserstein_valid_idx):
            distance_vec.append('wasserstein')
        if not np.any(bottleneck_valid_idx):
            distance_vec.append('bottleneck')
        return distance_vec

    return check_func


##### score viz #########################

# full pipeline results
def plot_full():
    with open('ICONES_Data/scores.pickle', 'rb') as fileH:
        scores_df = pickle.load(fileH)

    # scores_df = pd.read_json('ICONES_Data/scores.pickle')



    patch_size = np.unique(scores_df['patch_size'].values)
    epsilon_factor = np.unique(scores_df['epsilon_factor'].values)
    distance       = np.unique(scores_df['distance'].values)
    # uniform_affinity_operator = np.unique(scores_df['uniform_affinity_operator'].values)
    uniform_affinity_operator = [True]
    homology_dim = np.unique(scores_df['homology_dim'].values)

    # # plot best results for all runs
    # mAP_test = scores_df['mAP_test'].values
    # mAP_train = scores_df['mAP_train'].values
    #
    # idx_mAP_test = np.argsort(mAP_test)[::-1]
    # idx_mAP_train = np.argsort(mAP_train)[::-1]
    # test_op = (mAP_test[idx_mAP_test[0]], mAP_train[idx_mAP_test[0]])
    # train_op = (mAP_test[idx_mAP_train[0]], mAP_train[idx_mAP_train[0]])

    performance_criterion = ['mAP_test', 'mAP_val', 'mAP_train', 'DM_clusters_scores', 'metric_cluster_summary']
    rows_name = ['p%d' %ii for ii in patch_size]
    cols_name = ['e%s' %ii for ii in epsilon_factor]


    fig = [make_subplots(rows=2,
                         cols=2,
                         subplot_titles=("Bottleneck",
                                         "Wasserstein",
                                         "Bottleneck",
                                         "Wasserstein"),
                         shared_yaxes=True) for _ in range(12)]
    for dim in homology_dim:
        dim_valid_idx = scores_df['homology_dim'].values == dim
        for uniform_affinity in uniform_affinity_operator:
            uniform_valid_idx = scores_df['uniform_affinity_operator'].values == uniform_affinity
            for distance_type in distance:
                distance_valid_idx = scores_df['distance'].values == distance_type
                test_scores_table = np.zeros((epsilon_factor.shape[0], patch_size.shape[0], 2))
                train_scores_table = np.zeros((epsilon_factor.shape[0], patch_size.shape[0], 2))
                val_scores_table = np.zeros((epsilon_factor.shape[0], patch_size.shape[0], 2))
                train_val_scores_table = np.zeros((epsilon_factor.shape[0], patch_size.shape[0], 2))
                DM_clusters = np.zeros((epsilon_factor.shape[0], patch_size.shape[0], 3))
                kNN_clusters = np.zeros((epsilon_factor.shape[0], patch_size.shape[0]))
                for ii, patch in enumerate(patch_size):
                    patch_size_valid_idx = scores_df['patch_size'].values == patch
                    for jj, eps in enumerate(epsilon_factor):
                        epsilon_factor_valid_idx = scores_df['epsilon_factor'].values == eps
                        valid_idx = uniform_valid_idx & distance_valid_idx & patch_size_valid_idx & epsilon_factor_valid_idx & dim_valid_idx
                        if np.sum(valid_idx) > 1:
                            print("More then 1 instance %s %s %d %f" % (uniform_affinity, distance_type, patch, eps))
                        if np.sum(valid_idx) >= 1:
                            current_run_idx = np.where(valid_idx)[0][-1]
                            best_score_idx = np.argmax(np.array(scores_df.loc[current_run_idx]['mAP_test'])[:, :, 0])
                            best_score_idx -= np.sum(np.nonzero(np.array(scores_df.loc[current_run_idx]['mAP_test'])[:, :, 0].flatten() == 0) < best_score_idx)
                            best_row, best_col = np.unravel_index(best_score_idx, np.array(scores_df.loc[current_run_idx]['mAP_test'])[:, :, 0].shape)

                            test_scores_table[jj, ii, :] = np.array(scores_df.loc[current_run_idx]['mAP_test'])[best_row, best_col, :]
                            train_scores_table[jj, ii, :] = np.array(scores_df.loc[current_run_idx]['mAP_train'])[best_row, best_col, :]
                            val_scores_table[jj, ii, :] = np.array(scores_df.loc[current_run_idx]['mAP_val'])[best_row, best_col, :]
                            train_val_scores_table[jj, ii, :] = np.array(scores_df.loc[current_run_idx]['mAP_train_val'])[best_row, best_col, :]

                            DM_clusters[jj, ii, :] = np.array(scores_df.loc[current_run_idx]['DM_clusters_scores']).reshape((-1, 3)).max(axis=0)
                            kNN_clusters[jj, ii] = np.array(scores_df.loc[current_run_idx]['metric_cluster_summary']).max()
                        else:
                            test_scores_table[jj, ii, :]      = None
                            train_scores_table[jj, ii, :]     = None
                            val_scores_table[jj, ii, :]       = None
                            train_val_scores_table[jj, ii, :] = None
                            DM_clusters[jj, ii, :]            = None
                            kNN_clusters[jj, ii]              = None

                col_idx = (distance_type == 'wasserstein') * 1 + 1
                fig[0].add_trace(go.Heatmap(z=test_scores_table[:, :, 0],
                                            x=rows_name,
                                            y=cols_name,
                                            text=test_scores_table[:, :, 1],
                                            coloraxis="coloraxis"),
                                 row=int(dim + 1), col=col_idx)
                fig[1].add_trace(go.Heatmap(z=train_scores_table[:, :, 0],
                                            x=rows_name,
                                            y=cols_name,
                                            text=train_scores_table[:, :, 1],
                                            coloraxis="coloraxis"),
                                 row=int(dim + 1), col=col_idx)
                fig[2].add_trace(go.Heatmap(z=val_scores_table[:, :, 0],
                                            x=rows_name,
                                            y=cols_name,
                                            text=val_scores_table[:, :, 1],
                                            coloraxis="coloraxis"),
                                 row=int(dim + 1), col=col_idx)
                fig[3].add_trace(go.Heatmap(z=train_val_scores_table[:, :, 0],
                                            x=rows_name,
                                            y=cols_name,
                                            text=train_val_scores_table[:, :, 1],
                                            coloraxis="coloraxis"),
                                 row=int(dim + 1), col=col_idx)
                for eps_idx, eps in enumerate(cols_name):
                    fig[4].add_trace(go.Bar(name=eps,
                                            x=patch_size,
                                            y=test_scores_table[eps_idx, :, 0],
                                            error_y=dict(type='data', array=test_scores_table[eps_idx, :, 1])),
                                     row=int(dim + 1), col=col_idx)
                    fig[5].add_trace(go.Bar(name=eps,
                                            x=patch_size,
                                            y=train_scores_table[eps_idx, :, 0],
                                            error_y=dict(type='data', array=train_scores_table[eps_idx, :, 1])),
                                     row=int(dim + 1), col=col_idx)
                    fig[6].add_trace(go.Bar(name=eps,
                                            x=patch_size,
                                            y=val_scores_table[eps_idx, :, 0],
                                            error_y=dict(type='data', array=val_scores_table[eps_idx, :, 1])),
                                     row=int(dim + 1), col=col_idx)
                    fig[7].add_trace(go.Bar(name=eps,
                                            x=patch_size,
                                            y=train_val_scores_table[eps_idx, :, 0],
                                            error_y=dict(type='data', array=train_val_scores_table[eps_idx, :, 1])),
                                     row=int(dim + 1), col=col_idx)

                for index in range(3):
                    fig[8 + index].add_trace(go.Heatmap(z=DM_clusters[:, :, index],
                                                        x=rows_name,
                                                        y=cols_name,
                                                        coloraxis="coloraxis"),
                                            row=int(dim + 1), col=col_idx)
                fig[11].add_trace(go.Heatmap(z=kNN_clusters,
                                            x=rows_name,
                                            y=cols_name,
                                            coloraxis="coloraxis"),
                                 row=int(dim + 1), col=col_idx)


    fig[0].update_layout(title='Test Set',
                         coloraxis_colorbar=dict(title="mAP"))
    fig[1].update_layout(title='Train Set',
                         coloraxis_colorbar=dict(title="mAP"))
    fig[2].update_layout(title='Val Set',
                         coloraxis_colorbar=dict(title="mAP"))
    fig[3].update_layout(title='Train Val Set',
                         coloraxis_colorbar=dict(title="mAP"))
    fig[4].update_layout(title='Test Set')
    fig[5].update_layout(title='Train Set')
    fig[6].update_layout(title='Val Set')
    fig[7].update_layout(title='Train Val Set')
    fig[8].update_layout(title='DM Calinski Harabasz')
    fig[9].update_layout(title='DM Davies Bouldin')
    fig[10].update_layout(title='DM Silhouette')
    fig[11].update_layout(title='kNN score')

    for ii in range(4):
        fig[ii].data[0].update(zmin=0, zmax=1)
        fig[ii].update_yaxes(title_text="H_0", row=1, col=1)
        fig[ii].update_yaxes(title_text="H_1", row=2, col=1)

    for ii in range(4, 8):
        fig[ii].update_layout(yaxis=dict(range=[0, 1]))
        fig[ii].update_layout(yaxis5=dict(range=[0, 1]))

    fig[11].data[0].update(zmin=0, zmax=1)


    for ii in range(len(fig)):
        fig[ii].show()

        #############################################################################################
        ## plot train and test heat
    with open('ICONES_Data/scores.pickle', 'rb') as fileH:
        scores_df = pickle.load(fileH)

    # scores_df = pd.read_json('ICONES_Data/scores.pickle')

    patch_size = np.unique(scores_df['patch_size'].values)
    epsilon_factor = np.unique(scores_df['epsilon_factor'].values)
    rows_name = ['s%d' % ii for ii in patch_size]
    cols_name = ['e%s' % ii for ii in epsilon_factor]

    fig_train = go.Figure()
    fig_test = go.Figure()

    dim_valid_idx = scores_df['homology_dim'].values == 1
    uniform_valid_idx = scores_df['uniform_affinity_operator'].values == True
    distance_valid_idx = scores_df['distance'].values == 'wasserstein'
    test_scores_table = np.zeros((epsilon_factor.shape[0], patch_size.shape[0], 2))
    train_scores_table = np.zeros((epsilon_factor.shape[0], patch_size.shape[0], 2))
    for ii, patch in enumerate(patch_size):
        patch_size_valid_idx = scores_df['patch_size'].values == patch
        for jj, eps in enumerate(epsilon_factor):
            epsilon_factor_valid_idx = scores_df['epsilon_factor'].values == eps
            valid_idx = uniform_valid_idx & distance_valid_idx & patch_size_valid_idx & epsilon_factor_valid_idx & dim_valid_idx
            # if np.sum(valid_idx) > 1:
                # print("More then 1 instance %s %s %d %f" % (uniform_affinity, distance_type, patch, eps))
            if np.sum(valid_idx) >= 1:
                current_run_idx = np.where(valid_idx)[0][-1]
                best_score_idx = np.argmax(np.array(scores_df.loc[current_run_idx]['mAP_test'])[:, :, 0])
                best_score_idx -= np.sum(np.nonzero(np.array(scores_df.loc[current_run_idx]['mAP_test'])[:, :, 0].flatten() == 0) < best_score_idx)
                best_row, best_col = np.unravel_index(best_score_idx, np.array(scores_df.loc[current_run_idx]['mAP_test'])[:, :, 0].shape)

                test_scores_table[jj, ii, :] = np.array(scores_df.loc[current_run_idx]['mAP_test'])[best_row, best_col, :]
                train_scores_table[jj, ii, :] = np.array(scores_df.loc[current_run_idx]['mAP_train'])[best_row, best_col, :]

            else:
                test_scores_table[jj, ii, :]      = None
                train_scores_table[jj, ii, :]     = None

    fig_test.add_trace(go.Heatmap(z=test_scores_table[:, :, 0],
                                x=rows_name,
                                y=cols_name,
                                text=test_scores_table[:, :, 1],
                                colorbar=dict(title="mAP"),
                                zmin=0,
                                zmax=1))
    fig_train.add_trace(go.Heatmap(z=train_scores_table[:, :, 0],
                                x=rows_name,
                                y=cols_name,
                                text=train_scores_table[:, :, 1],
                               colorbar=dict(title="mAP"),
                               zmin=0,
                               zmax=1))

    fig_test.update_layout(coloraxis_colorbar=dict(title="mAP"),
                           font=dict(size=22,
                                     color='black'),
                           width=900,
                           height=500,
                           xaxis_title="Patch Size",
                           yaxis_title="Kernel Scale",
                           )
    fig_train.update_layout(coloraxis_colorbar=dict(title="mAP"),
                            font=dict(size=22,
                                      color='black'),
                            width=900,
                            height=500,
                            xaxis_title="Patch Size",
                            yaxis_title="Kernel Scale",
                            )
    fig_test.show()
    fig_train.show()
    fig_test.write_image('heat_test.png')
    fig_train.write_image('heat_train.png')



# geometry results
def plot_geometry():

    scores_geometry_df = pd.read_json('ICONES_Data/scores_geometry.json')

    patch_size = np.unique(scores_geometry_df['patch_size'].values)
    epsilon_factor = np.unique(scores_geometry_df['epsilon_factor'].values)
    rows_name = ['p%d' %ii for ii in patch_size]
    cols_name = ['e%s' %ii for ii in epsilon_factor]

    fig = make_subplots(rows=2,
                        cols=4,
                        subplot_titles=("Test set",
                                        "Train set",
                                        "Val set",
                                        "Train Val set",
                                        "Test set",
                                        "Train set",
                                        "Val set",
                                        "Train Val set"),
                        shared_yaxes=True)
    fig_clusters = [go.Figure() for _ in range(4)]
    test_scores_table = np.zeros((epsilon_factor.shape[0], patch_size.shape[0], 2))
    train_scores_table = np.zeros((epsilon_factor.shape[0], patch_size.shape[0], 2))
    val_scores_table = np.zeros((epsilon_factor.shape[0], patch_size.shape[0], 2))
    train_val_scores_table = np.zeros((epsilon_factor.shape[0], patch_size.shape[0], 2))
    DM_clusters = np.zeros((epsilon_factor.shape[0], patch_size.shape[0], 3))
    kNN_clusters = np.zeros((epsilon_factor.shape[0], patch_size.shape[0]))
    for ii, patch in enumerate(patch_size):
        patch_size_valid_idx = scores_geometry_df['patch_size'].values == patch
        for jj, eps in enumerate(epsilon_factor):
            epsilon_factor_valid_idx = scores_geometry_df['epsilon_factor'].values == eps
            valid_idx = epsilon_factor_valid_idx & patch_size_valid_idx
            if np.sum(valid_idx) > 1:
                print("More then 1 instance %f %d" % (eps, patch))
            if np.sum(valid_idx) >= 1:
                current_run_idx = np.where(valid_idx)[0][-1]

                best_score_idx = np.argmax(np.array(scores_geometry_df.loc[current_run_idx]['mAP_test'])[:, :, 0])
                best_score_idx -= np.sum(np.nonzero(
                    np.array(scores_geometry_df.loc[current_run_idx]['mAP_test'])[:, :, 0].flatten() == 0) < best_score_idx)
                best_row, best_col = np.unravel_index(best_score_idx,
                                                      np.array(scores_geometry_df.loc[current_run_idx]['mAP_test'])[:, :, 0].shape)

                test_scores_table[jj, ii, :] = np.array(scores_geometry_df.loc[current_run_idx]['mAP_test'])[best_row, best_col, :]
                train_scores_table[jj, ii, :] = np.array(scores_geometry_df.loc[current_run_idx]['mAP_train'])[best_row, best_col, :]
                val_scores_table[jj, ii, :] = np.array(scores_geometry_df.loc[current_run_idx]['mAP_val'])[best_row, best_col, :]
                train_val_scores_table[jj, ii, :] = np.array(scores_geometry_df.loc[current_run_idx]['mAP_train_val'])[best_row, best_col, :]

                DM_clusters[jj, ii, :] = np.array(scores_geometry_df.loc[current_run_idx]['DM_clusters_scores']).reshape((-1, 3)).max(axis=0)
                kNN_clusters[jj, ii] = np.array(scores_geometry_df.loc[current_run_idx]['metric_cluster_summary']).max()
            else:
                test_scores_table[jj, ii, :] = None
                train_scores_table[jj, ii, :] = None
                val_scores_table[jj, ii, :] = None
                train_val_scores_table[jj, ii, :] = None
                DM_clusters[jj, ii, :] = None
                kNN_clusters[jj, ii] = None

    fig.add_trace(go.Heatmap(z=test_scores_table[:, :, 0],
                             x=rows_name,
                             y=cols_name,
                             text=test_scores_table[:, :, 1],
                             coloraxis="coloraxis",
                             zmin=0,
                             zmax=1),
                     row=1, col=1)
    fig.add_trace(go.Heatmap(z=train_scores_table[:, :, 0],
                             x=rows_name,
                             y=cols_name,
                             text=train_scores_table[:, :, 1],
                             coloraxis="coloraxis"),
                     row=1, col=2)
    fig.add_trace(go.Heatmap(z=val_scores_table[:, :, 0],
                             x=rows_name,
                             y=cols_name,
                             text=val_scores_table[:, :, 1],
                             coloraxis="coloraxis"),
                  row=1, col=3)
    fig.add_trace(go.Heatmap(z=train_val_scores_table[:, :, 0],
                             x=rows_name,
                             y=cols_name,
                             text=train_val_scores_table[:, :, 1],
                             coloraxis="coloraxis"),
                  row=1, col=4)
    for eps_idx, eps in enumerate(cols_name):
        fig.add_trace(go.Bar(name=eps,
                                x=patch_size,
                                y=test_scores_table[eps_idx, :, 0],
                                error_y=dict(type='data', array=test_scores_table[eps_idx, :, 1]),
                             showlegend=False),
                         row=2, col=1)
        fig.add_trace(go.Bar(name=eps,
                             x=patch_size,
                             y=train_scores_table[eps_idx, :, 0],
                             error_y=dict(type='data', array=train_scores_table[eps_idx, :, 1]),
                             showlegend=False),
                         row=2, col=2)
        fig.add_trace(go.Bar(name=eps,
                             x=patch_size,
                             y=val_scores_table[eps_idx, :, 0],
                             error_y=dict(type='data', array=val_scores_table[eps_idx, :, 1]),
                             showlegend=False),
                      row=2, col=3)
        fig.add_trace(go.Bar(name=eps,
                             x=patch_size,
                             y=train_val_scores_table[eps_idx, :, 0],
                             error_y=dict(type='data', array=train_val_scores_table[eps_idx, :, 1]),
                             showlegend=False),
                      row=2, col=4)
    for index in range(3):
        fig_clusters[index].add_trace(go.Heatmap(z=DM_clusters[:, :, index],
                                                 x=rows_name,
                                                 y=cols_name,
                                                 coloraxis="coloraxis"))
    fig_clusters[3].add_trace(go.Heatmap(z=kNN_clusters,
                                         x=rows_name,
                                         y=cols_name,
                                         coloraxis="coloraxis"))

    fig.update_layout(title='Geometry classification',
                         coloraxis_colorbar=dict(title="mAP"))
    fig_clusters[0].update_layout(title='Geometry DM Calinski Harabasz')
    fig_clusters[1].update_layout(title='Geometry DM Davies Bouldin')
    fig_clusters[2].update_layout(title='Geometry DM Silhouette')
    fig_clusters[3].update_layout(title='Geometry kNN score')

    fig.data[0].update(zmin=0, zmax=1)
    fig.data[1].update(zmin=0, zmax=1)
    fig.update_layout(yaxis3=dict(range=[0, 1]))
    fig_clusters[3].data[0].update(zmin=0, zmax=1)

    fig.show()
    fig_clusters[0].show()
    fig_clusters[1].show()
    fig_clusters[2].show()
    fig_clusters[3].show()


def plot_homology():
    # homology results

    scores_homology_df = pd.read_json('ICONES_Data/scores_homology.json')

    patch_size = np.unique(scores_homology_df['patch_size'].values)
    distance       = np.unique(scores_homology_df['distance'].values)
    homology_dim = np.unique(scores_homology_df['homology_dim'].values)

    rows_name = ['p%d' %ii for ii in patch_size]
    cols_name = ['%s' %ii for ii in distance]

    fig = [make_subplots(rows=2,
                         cols=4,
                         subplot_titles=("Test set",
                                        "Train set",
                                        "Val set",
                                        "Train Val set",
                                        "Test set",
                                        "Train set",
                                        "Val set",
                                        "Train Val set"),
                         shared_yaxes=True) for _ in range(2)]
    fig_clusters = [make_subplots(rows=2,
                                  cols=1,
                                  subplot_titles=("H_0",
                                                  "H_1")) for _ in range(4)]
    for dim in homology_dim:
        dim_valid_idx = scores_homology_df['homology_dim'].values == dim
        test_scores_table = np.zeros((distance.shape[0], patch_size.shape[0], 2))
        train_scores_table = np.zeros((distance.shape[0], patch_size.shape[0], 2))
        val_scores_table = np.zeros((distance.shape[0], patch_size.shape[0], 2))
        train_val_scores_table = np.zeros((distance.shape[0], patch_size.shape[0], 2))
        DM_clusters = np.zeros((distance.shape[0], patch_size.shape[0], 3))
        kNN_clusters = np.zeros((distance.shape[0], patch_size.shape[0]))
        for ii, patch in enumerate(patch_size):
            patch_size_valid_idx = scores_homology_df['patch_size'].values == patch
            for jj, distance_type in enumerate(distance):
                distance_valid_idx = scores_homology_df['distance'].values == distance_type
                valid_idx = distance_valid_idx & patch_size_valid_idx & dim_valid_idx
                if np.sum(valid_idx) > 1:
                    print("More then 1 instance %s %d" % (distance_type, patch))
                if np.sum(valid_idx) >= 1:
                    current_run_idx = np.where(valid_idx)[0][-1]

                    best_score_idx = np.argmax(np.array(scores_homology_df.loc[current_run_idx]['mAP_test'])[:, :, 0])
                    best_score_idx -= np.sum(np.nonzero(np.array(scores_homology_df.loc[current_run_idx]['mAP_test']).flatten() == 0) < best_score_idx)
                    best_row, best_col = np.unravel_index(best_score_idx,
                                                          np.array(scores_homology_df.loc[current_run_idx]['mAP_test'])[:, :, 0].shape)

                    test_scores_table[jj, ii, :] = np.array(scores_homology_df.loc[current_run_idx]['mAP_test'])[best_row, best_col, :]
                    train_scores_table[jj, ii, :] = np.array(scores_homology_df.loc[current_run_idx]['mAP_train'])[ best_row, best_col, :]
                    val_scores_table[jj, ii, :] = np.array(scores_homology_df.loc[current_run_idx]['mAP_val'])[best_row, best_col, :]
                    train_val_scores_table[jj, ii, :] = np.array(scores_homology_df.loc[current_run_idx]['mAP_train_val'])[best_row, best_col, :]

                    DM_clusters[jj, ii, :] = np.array(scores_homology_df.loc[current_run_idx]['DM_clusters_scores']).reshape(
                        (-1, 3)).max(axis=0)
                    kNN_clusters[jj, ii] = np.array(scores_homology_df.loc[current_run_idx]['metric_cluster_summary']).max()
                else:
                    test_scores_table[jj, ii, :] = None
                    train_scores_table[jj, ii, :] = None
                    val_scores_table[jj, ii, :] = None
                    train_val_scores_table[jj, ii, :] = None
                    DM_clusters[jj, ii, :] = None
                    kNN_clusters[jj, ii] = None

        fig[0].add_trace(go.Heatmap(z=test_scores_table[:, :, 0],
                                    x=rows_name,
                                    y=cols_name,
                                    text=test_scores_table[:, :, 1],
                                    coloraxis="coloraxis",
                                    zmin=0,
                                    zmax=1),
                         row=int(1 + dim), col=1)
        fig[0].add_trace(go.Heatmap(z=train_scores_table[:, :, 0],
                                    x=rows_name,
                                    y=cols_name,
                                    text=train_scores_table[:, :, 1],
                                    coloraxis="coloraxis",
                                    zmin=0,
                                    zmax=1),
                         row=int(1 + dim), col=2)
        fig[0].add_trace(go.Heatmap(z=val_scores_table[:, :, 0],
                                    x=rows_name,
                                    y=cols_name,
                                    text=val_scores_table[:, :, 1],
                                    coloraxis="coloraxis",
                                    zmin=0,
                                    zmax=1),
                         row=int(1 + dim), col=3)
        fig[0].add_trace(go.Heatmap(z=train_val_scores_table[:, :, 0],
                                    x=rows_name,
                                    y=cols_name,
                                    text=train_val_scores_table[:, :, 1],
                                    coloraxis="coloraxis",
                                    zmin=0,
                                    zmax=1),
                         row=int(1 + dim), col=4)
        for dist_idx, dist in enumerate(distance):
            fig[1].add_trace(go.Bar(name=dist,
                                 x=patch_size,
                                 y=test_scores_table[dist_idx, :, 0],
                                 error_y=dict(type='data', array=test_scores_table[dist_idx, :, 1])),
                          row=int(1 + dim), col=1)
            fig[1].add_trace(go.Bar(name=dist,
                                 x=patch_size,
                                 y=train_scores_table[dist_idx, :, 0],
                                 error_y=dict(type='data', array=train_scores_table[dist_idx, :, 1])),
                             row=int(1 + dim), col=2)
            fig[1].add_trace(go.Bar(name=dist,
                                    x=patch_size,
                                    y=val_scores_table[dist_idx, :, 0],
                                    error_y=dict(type='data', array=val_scores_table[dist_idx, :, 1])),
                             row=int(1 + dim), col=3)
            fig[1].add_trace(go.Bar(name=dist,
                                    x=patch_size,
                                    y=train_val_scores_table[dist_idx, :, 0],
                                    error_y=dict(type='data', array=train_val_scores_table[dist_idx, :, 1])),
                             row=int(1 + dim), col=4)
        for index in range(3):
            fig_clusters[index].add_trace(go.Heatmap(z=DM_clusters[:, :, index],
                                                x=rows_name,
                                                y=cols_name,
                                                coloraxis="coloraxis2"),
                                          row=int(1 + dim), col=1)
        fig_clusters[3].add_trace(go.Heatmap(z=kNN_clusters,
                                    x=rows_name,
                                    y=cols_name,
                                    coloraxis="coloraxis",
                                    zmin=0,
                                    zmax=1),
                                  row=int(1 + dim), col=1)

    fig[0].update_layout(title='Homology classification',
                         coloraxis_colorbar=dict(title="mAP"))
    fig[1].update_layout(title='Homology classification')
    fig_clusters[0].update_layout(title='Homology DM Calinski Harabasz')
    fig_clusters[1].update_layout(title='Homology DM Davies Bouldin')
    fig_clusters[2].update_layout(title='Homology DM Silhouette')
    fig_clusters[3].update_layout(title='Homology kNN score')

    fig[0].data[0].update(zmin=0, zmax=1)
    fig[0].update_yaxes(title_text="H_0", row=1, col=1)
    fig[0].update_yaxes(title_text="H_1", row=2, col=1)
    fig[1].update_layout(yaxis=dict(range=[0, 1]))
    fig[1].update_layout(yaxis3=dict(range=[0, 1]))
    fig_clusters[3].data[0].update(zmin=0, zmax=1)
    fig_clusters[3].data[1].update(zmin=0, zmax=1)


    fig[0].show()
    fig[1].show()
    fig_clusters[0].show()
    fig_clusters[1].show()
    fig_clusters[2].show()
    fig_clusters[3].show()


# baseline
def plot_comparison():
    # hand pick from above functions
    baseline_classification = np.array([[0.41, 0.84],
                                        [0.047, 0.017]])
    geometry_classification = np.array([[0.59, 0.787],
                                    [0.056, 0.082]])
    homology_classification = np.array([[0.673, 0.833],
                                    [0.079, 0.008]])
    full_classification = np.array([[0.813, 0.991],
                                    [0.026, 0.009]])
    kNN_score = np.array([0.189, 0.434, 0.51, 0.73])


    # plot summery of all oblation
    algorithm_name = ['Baseline', 'Only Geometry', 'Only TDA', 'Ours']


    fig = go.Figure(data=[
        go.Bar(name='Test',
               x=algorithm_name,
               y=[baseline_classification[0][0],
                  geometry_classification[0][0],
                  homology_classification[0][0],
                  full_classification[0][0]],
               error_y=dict(type='data', array=[baseline_classification[1][0],
                                                geometry_classification[1][0],
                                                homology_classification[1][0],
                                                full_classification[1][0]]),
               textposition='auto'),
        go.Bar(name='Train',
               x=algorithm_name,
               y=[baseline_classification[0][1],
                  geometry_classification[0][1],
                  homology_classification[0][1],
                  full_classification[0][1]],
               error_y=dict(type='data', array=[baseline_classification[1][1],
                                                geometry_classification[1][1],
                                                homology_classification[1][1],
                                                full_classification[1][1]]),
               textposition='auto'),
    ])
    # Change the bar mode
    fig.update_layout(barmode='group',
                      font=dict(size=20,
                                color='black'),
                      yaxis_title="mAP",
                      width = 800,
                      height = 400)
    fig.update_yaxes(range=[0, 1])
    fig.show()
    fig.write_image('Ablation_study.png')


def plot_cm(scores_df, patch, eps, uniform_affinity, distance_type, dim,
            DM_uniform, DM_eps,
            svm_c, svm_scale, svm_n_features):

    dim_valid_idx = scores_df['homology_dim'].values == dim
    uniform_valid_idx = scores_df['uniform_affinity_operator'].values == uniform_affinity
    distance_valid_idx = scores_df['distance'].values == distance_type
    patch_size_valid_idx = scores_df['patch_size'].values == patch
    epsilon_factor_valid_idx = scores_df['epsilon_factor'].values == eps
    valid_idx = uniform_valid_idx & distance_valid_idx & patch_size_valid_idx & epsilon_factor_valid_idx & dim_valid_idx

    current_run_idx = np.where(valid_idx)[0][-1]
    file_path = scores_df.loc[current_run_idx]['file']

    with open("../%s" % file_path, 'rb') as fileH:
        mat_dict = pickle.load(fileH)

    DM_embedding = DR.DiffusionMaps(mat_dict["mat_dist"][dim],
                                    epsilon_factor=DM_eps,
                                    uniformity=DM_uniform)

    embedding = DM_embedding['embedding']
    scaler = StandardScaler()
    embedding = scaler.fit_transform(embedding)
    with open('../ICONES_Data/ICONES-HSI/labels.pickle', 'rb') as file:
        labels = pickle.load(file)

    n_samples = 486
    kfold_n_splits = 10

    one_hot_labels = label_binarize(labels, classes=np.unique(labels))
    embedding = scaler.fit_transform(embedding)

    # SVM
    n_labels_in_class = np.sum(one_hot_labels, axis=0)
    class_weight = n_labels_in_class / n_samples
    class_weight = {ii: value for ii, value in enumerate(class_weight)}

    skf = StratifiedKFold(n_splits=kfold_n_splits, shuffle=True, random_state=0)

    multiclass_svm = SVC(C=svm_c,
                         gamma=svm_scale,
                         decision_function_shape='ovr',
                         probability=True,
                         class_weight=class_weight)

    train_cm = np.zeros((np.unique(labels).shape[0], np.unique(labels).shape[0], kfold_n_splits))
    test_cm = np.zeros((np.unique(labels).shape[0], np.unique(labels).shape[0], kfold_n_splits))
    mAP_test = np.zeros(kfold_n_splits)
    mAP_train = np.zeros(kfold_n_splits)

    best_classifier = multiclass_svm
    # train and test confusion matrix
    iter = 0
    for train_index, test_index in skf.split(embedding, labels):
        best_classifier.fit(embedding[train_index, 0:svm_n_features], labels[train_index])
        train_pred = best_classifier.predict(embedding[train_index, 0:svm_n_features])
        test_pred = best_classifier.predict(embedding[test_index, 0:svm_n_features])
        train_prob = best_classifier.predict_proba(embedding[train_index, 0:svm_n_features])
        test_prob = best_classifier.predict_proba(embedding[test_index, 0:svm_n_features])
        mAP_train[iter] = average_precision_score(label_binarize(labels[train_index], classes=np.unique(labels)), train_prob, average='micro')
        mAP_test[iter] = average_precision_score(label_binarize(labels[test_index], classes=np.unique(labels)), test_prob, average='micro')
        train_cm[:, :, iter] = confusion_matrix(labels[train_index], train_pred, labels=np.unique(labels))
        train_cm[:, :, iter] /= np.sum(train_cm[:, :, iter], axis=1, keepdims=True)
        test_cm[:, :, iter] = confusion_matrix(labels[test_index], test_pred, labels=np.unique(labels))
        test_cm[:, :, iter] /= np.sum(test_cm[:, :, iter], axis=1, keepdims=True)
        iter += 1

    mean_train_cm = np.mean(train_cm, axis=-1)
    std_train_cm = np.std(train_cm, axis=-1)
    mean_test_cm = np.mean(test_cm, axis=-1)
    std_test_cm = np.std(test_cm, axis=-1)

    x = labels_names
    y = labels_names

    z_text = [["%.2f +- %.2f" % (m, s) for m, s in zip(mm, ss)] for mm, ss in zip(mean_train_cm, std_train_cm)]
    fig_train = ff.create_annotated_heatmap(mean_train_cm, x=x, y=y, annotation_text=z_text, colorscale='Viridis')
    fig_train.update_layout(title_text='<i><b>Train Confusion matrix - mAP:%.2f +- %.2f</b></i>' % (np.mean(mAP_train),
                                                                                                    np.std(mAP_train)))
    fig_train.add_annotation(dict(font=dict(color="black", size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))
    fig_train.add_annotation(dict(font=dict(color="black", size=14),
                            x=-0.35,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))


    fig_train.update_layout(
        xaxis_title="Predicted value",
        yaxis_title="Real value",
        xaxis=dict(autorange='reversed'))

    fig_train['data'][0]['showscale'] = True
    z_text = [["%.2f +- %.2f" % (m, s) for m, s in zip(mm, ss)] for mm, ss in zip(mean_test_cm, std_test_cm)]
    fig_test = ff.create_annotated_heatmap(mean_test_cm, x=x, y=y, annotation_text=z_text, colorscale='Viridis')
    fig_test.update_layout(title_text='<i><b>Test Confusion matrix mAP: %.2f +- %.2f</b></i>' % (np.mean(mAP_test),
                                                                                                 np.std(mAP_test)))
    fig_test.add_annotation(dict(font=dict(color="black", size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))

    fig_test.add_annotation(dict(font=dict(color="black", size=14),
                            x=-0.35,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    fig_test.update_layout(
        xaxis_title="Predicted value",
        yaxis_title="Real value",
        xaxis=dict(autorange='reversed'))

    fig_test['data'][0]['showscale'] = True
    return fig_train, fig_test


def plot_tSNE(scores_df, patch, eps, uniform_affinity, distance_type, dim,
                      DM_uniform, DM_eps,
                      p, lr, early_exaggeration, metric, vis_n_features):

    dim_valid_idx = scores_df['homology_dim'].values == dim
    uniform_valid_idx = scores_df['uniform_affinity_operator'].values == uniform_affinity
    distance_valid_idx = scores_df['distance'].values == distance_type
    patch_size_valid_idx = scores_df['patch_size'].values == patch
    epsilon_factor_valid_idx = scores_df['epsilon_factor'].values == eps
    valid_idx = uniform_valid_idx & distance_valid_idx & patch_size_valid_idx & epsilon_factor_valid_idx & dim_valid_idx

    current_run_idx = np.where(valid_idx)[0][0]
    file_path = scores_df.loc[current_run_idx]['file']

    with open("../%s" % file_path, 'rb') as fileH:
        mat_dict = pickle.load(fileH)
    with open('../ICONES_Data/ICONES-HSI/labels.pickle', 'rb') as file:
            labels = pickle.load(file)
    colormap = px.colors.qualitative.Set1
    raw_symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'octagon', 'star', 'hourglass']

    if metric == 'metric':
        embedding = mat_dict["mat_dist"][0]
        tSNE_embedding = TSNE(n_components=2,
                              perplexity=p,
                              learning_rate=lr,
                              early_exaggeration=early_exaggeration,
                              n_iter=2000,
                              metric='precomputed').fit_transform(embedding)

    else:
        DM_embedding = DR.DiffusionMaps(mat_dict["mat_dist"][0],
                                        epsilon_factor=DM_eps,
                                        uniformity=DM_uniform)
        embedding = DM_embedding['embedding']
        scaler = StandardScaler()
        embedding = scaler.fit_transform(embedding)
        tSNE_embedding = TSNE(n_components=2,
                              perplexity=p,
                              learning_rate=lr,
                              early_exaggeration=early_exaggeration,
                              n_iter=2000).fit_transform(embedding[:, 0:vis_n_features])

    fig_tSNE = go.Figure()
    for idx, label in enumerate(labels_names):
        fig_tSNE.add_trace(go.Scatter(x=tSNE_embedding[labels == idx, 0],
                                      y=tSNE_embedding[labels == idx, 1],
                                      marker_symbol=raw_symbols[idx],
                                      mode='markers',
                                      marker=dict(color=idx,
                                                  size=12,
                                                  colorscale=colormap),
                                      showlegend=True,
                                      name=label,
                                      text=labels))
    fig_tSNE.update_layout(font=dict(size=40,
                                    color='black'),
                           height=1100,
                           width=1300,
                           xaxis_title="tSNE Axis 1",
                           yaxis_title="tSNE Axis 2")
    fig_tSNE.write_image('HSI_tsne.jpg')
    return fig_tSNE

def plot_UMAP(scores_df, patch, eps, uniform_affinity, distance_type, dim,
                      DM_uniform, DM_eps,
                      min_dist, n_neighbores_umap, metric, vis_n_features):

    dim_valid_idx = scores_df['homology_dim'].values == dim
    uniform_valid_idx = scores_df['uniform_affinity_operator'].values == uniform_affinity
    distance_valid_idx = scores_df['distance'].values == distance_type
    patch_size_valid_idx = scores_df['patch_size'].values == patch
    epsilon_factor_valid_idx = scores_df['epsilon_factor'].values == eps
    valid_idx = uniform_valid_idx & distance_valid_idx & patch_size_valid_idx & epsilon_factor_valid_idx & dim_valid_idx

    current_run_idx = np.where(valid_idx)[0][0]
    file_path = scores_df.loc[current_run_idx]['file']

    with open("../%s" % file_path, 'rb') as fileH:
        mat_dict = pickle.load(fileH)
    with open('../ICONES_Data/ICONES-HSI/labels.pickle', 'rb') as file:
            labels = pickle.load(file)
    colormap = px.colors.qualitative.Set1
    raw_symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'octagon', 'star', 'hourglass']

    if metric == 'metric':
        embedding = mat_dict["mat_dist"][0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            uMAP_embedding = umap.UMAP(n_components=2,
                                       n_neighbors=n_neighbores_umap,
                                       min_dist=min_dist,
                                       metric='precomputed').fit_transform(embedding)
    else:
        DM_embedding = DR.DiffusionMaps(mat_dict["mat_dist"][0],
                                        epsilon_factor=DM_eps,
                                        uniformity=DM_uniform)
        embedding = DM_embedding['embedding']
        scaler = StandardScaler()
        embedding = scaler.fit_transform(embedding)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            uMAP_embedding = umap.UMAP(n_components=2,
                                       n_neighbors=n_neighbores_umap,
                                       min_dist=min_dist,
                                       metric='euclidean').fit_transform(embedding[:, 0:vis_n_features])

    fig_uMAP = go.Figure()
    for idx, label in enumerate(labels_names):
        fig_uMAP.add_trace(go.Scatter(x=uMAP_embedding[labels == idx, 0],
                                      y=uMAP_embedding[labels == idx, 1],
                                      marker_symbol=raw_symbols[idx],
                                      mode='markers',
                                      marker=dict(color=idx,
                                                  size=12,
                                                  colorscale=colormap),
                                      showlegend=True,
                                      name=label,
                                      text=labels))
    fig_uMAP.update_layout(title_text='UMAP',
                           height=1000,
                           width=1000)

    return fig_uMAP
#
