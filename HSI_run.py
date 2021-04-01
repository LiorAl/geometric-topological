import logging
import Utils.utils as utils
import os
import multiprocessing
from HSI_Data import HSI_Data
from HSI_utils.HSI_score_evaluation import evaluation
import HSI_utils.HSI_vizualization_results as viz
import numpy as np
import plotly.io as pio
import pickle
import glob
import pandas as pd
from HSI_utils.PCA_SVM_Classifier import PCA_SVM
from tqdm import tqdm

pio.renderers.default = "browser"

# np.random.seed(42)

# Script parts to run
generate_distance_matrix_run = False
analysis_run = False
baseline_run = False
geometry_run = False
homology_run = False
plot_results = True


def get_configs():
    conf = {
        'dataset_results_dir'      : 'Results/ICONES',
        'matrix_distance_dir'      : 'ICONES_Data/Distance_matrix',
        'run_result_dir'           : None,
        'kernel_distance'          : 'euclidean',  # seuclidean euclidean EMD
        'weighting_method'         : 'addition',  # direct  symmetric  eigenvalues addition subtraction
        'operator_norm'            : 'fro',
        'weight_func'              : 'Inverse',  # Inverse  NormalizeInverse  LogInverse  Identity ExponentialInverse
        'sample_process'           : None,  # None covariance
        'low_of_three'             : True,
        # data class flags and parameters
        'load_precompute_embedding': False,
        'init_data'                : True,
        'save_embedding'           : True,
        'plot_embedding'           : True,
        'save_figures'             : True,
        'beta_dim'                 : [0, 1],
        'visualize_data'           : False,
        'split_dataset'            : False,  # split each instance to 4 images
        'ignore_subjects'          : [],
        'run_baseline'             : False,

        # grid search hyperparams:
        'diagram_distance'         : ['wasserstein', 'bottleneck'],  # bottleneck wasserstein
        'patch_size'               : [50, 45, 40, 35, 30, 25, 20, 15, 10, 5],
        'base_epsilon_factor'      : [0.01, 0.1, 0.5, 1, 2, 5],
        'stride'                   : 1,  # in percentage (%) how much overlapping in patches
        'graph_type'               : ['spatial'],  # spectral spatial
        'diagram_type'             : 'diffusion_kernel_neighbors',
        'uniform_affinity_operator': True,  # uniformity affinity operator for uniform distribution
        'clean_small_homology'     : {"5": 0.1,
                                      "10": 0.2,
                                      "15": 1/3,
                                      "20": 0.5,
                                      "25": 0.8}, # or None
        'k_fold_split'             : 10,
    }
    return conf


if __name__ == "__main__":
    # load configs dict and pre-run procedure (logger and results dir load data)
    conf = get_configs()
    conf['run_result_dir'] = utils.PrepareResultsDir(conf['dataset_results_dir'],
                                                     debug=True)
    utils.set_logger(os.path.join(conf['run_result_dir'], 'log.log'))
    utils.save_dict_to_json(conf, os.path.join(conf['run_result_dir'], 'configs.json'))

    # run algorithm flow with selected configs
    if generate_distance_matrix_run:
        # get prepaid results
        check_func = viz.missing_param_run_function()

        matrix_counter = 0
        for hh, graph_type in enumerate(conf["graph_type"]):
            for ii, patch_size in enumerate(conf["patch_size"]):
                for jj, epsilon_factor in enumerate(conf['base_epsilon_factor']):
                    distance_vec = check_func(conf['uniform_affinity_operator'], patch_size, epsilon_factor)
                    if distance_vec:
                        DataObj = HSI_Data(conf, patch_size, epsilon_factor, graph_type)

                        for DiagramDistance_i in distance_vec:
                            logging.info("START: graph_type: %s patch size: %d  |  eps: %.2f  | distance: %s "
                                         % (graph_type, patch_size, epsilon_factor, DiagramDistance_i))
                            if patch_size <= 25:
                                clean_factor = conf['clean_small_homology'][str(patch_size)]
                            else:
                                clean_factor = None
                            DataObj.GenerateDiagDistance(DiagramDistance_i,
                                                         matrix_counter,
                                                         clean_factor)

                            dict_to_save = {'graph_type': graph_type,
                                            'patch_size': patch_size,
                                            'epsilon_factor': epsilon_factor,
                                            'distance': DiagramDistance_i,
                                            'uniform_affinity_operator': conf['uniform_affinity_operator'],
                                            'clean_small_homology': conf['clean_small_homology'],
                                            'mat_dist': DataObj.matDist}
                            if conf['save_embedding']:
                                while os.path.exists("%s/%d.pickle" % (conf['matrix_distance_dir'], matrix_counter)):
                                    matrix_counter += 1
                                with open("%s/%d.pickle" % (conf['matrix_distance_dir'], matrix_counter),
                                          'wb') as fileH:
                                    pickle.dump(dict_to_save, fileH)
                            matrix_counter += 1

    # load HSI score instance
    with open('ICONES_Data/ICONES-HSI/labels.pickle', 'rb') as file:
        labels = pickle.load(file)
    HSI_score = evaluation(labels)

    param_score_df = pd.DataFrame(columns=['file', 'graph_type', 'patch_size', 'epsilon_factor',
                                           'distance', 'homology_dim', 'uniform_affinity_operator',
                                           'mAP_test', 'mAP_val', 'mAP_train', 'mAP_train_val', 'svm_config',
                                           'DM_clusters_scores', 'DM_features_clusters', 'metric_cluster_summary'])

    # Analysis run - performance of all flow
    if analysis_run:
        logging.info('START: Classification')
        try:
            with open('ICONES_Data/scores.pickle', 'rb') as fileH:
                param_score_df = pickle.load(fileH)
            files_list = param_score_df['file'].values
            files_list = [file.replace("\\", "/") for file in files_list]
        except:
            files_list = []

        dist_mat_dict_files = glob.glob('%s/*.pickle' % (conf['matrix_distance_dir']))
        matrix_counter = len(param_score_df)
        for file_path in tqdm(dist_mat_dict_files):
            logging.info("File: %s" % file_path)
            with open(file_path, 'rb') as fileH:
                try:
                    mat_dict = pickle.load(fileH)
                    if not mat_dict['uniform_affinity_operator']:
                        continue
                except:
                    logging.info("Bad pickle file")

                else:
                    # Classification and clustering score
                    if file_path.replace("\\", "/") not in files_list:
                        for dim in range(len(mat_dict["mat_dist"])):
                            svm_test_set_mAP, svm_train_set_mAP, svm_val_set_mAP, svm_train_val_set_mAP, svm_config, \
                            DM_clusters_scores, DM_features_clusters, \
                            metric_cluster_summary = HSI_score.Run(mat_dict["mat_dist"][dim],
                                                                   classification=True,
                                                                   clustering=True,
                                                                   clustering_DM=False)
                            param_score_df.loc[matrix_counter] = {'file'       : file_path,
                                                   'graph_type'                : mat_dict['graph_type'],
                                                   'patch_size'                : mat_dict['patch_size'],
                                                   'epsilon_factor'            : mat_dict['epsilon_factor'],
                                                   'distance'                  : mat_dict['distance'],
                                                   'homology_dim'              : dim,
                                                   'uniform_affinity_operator' : mat_dict['uniform_affinity_operator'],
                                                   'mAP_test': svm_test_set_mAP,
                                                   'mAP_val': svm_val_set_mAP,
                                                   'mAP_train': svm_train_set_mAP,
                                                   'mAP_train_val': svm_train_val_set_mAP,
                                                   'svm_config' : svm_config,
                                                   'DM_clusters_scores' : DM_clusters_scores,
                                                   'DM_features_clusters' : DM_features_clusters,
                                                   'metric_cluster_summary' : metric_cluster_summary}
                            matrix_counter += 1
                        files_list.append(file_path.replace("\\", "/"))
                    with open('ICONES_Data/scores.pickle', 'wb') as fileH:
                        pickle.dump(param_score_df, fileH)


    if baseline_run:
        logging.info('Run Baseline: PCA-PCA-SVM')
        PCA_SVM()

    # Run and evaluate performance only of geometric / homology part

    try:
        if geometry_run :
            logging.info('Run Geometry oblation study')
            with open('ICONES_Data/scores_geometry.pickle', 'rb') as fileH:
                param_score_df = pickle.load(fileH)
        if homology_run:
            logging.info('Run Homology oblation study')
            with open('ICONES_Data/scores_homology.pickle', 'rb') as fileH:
                param_score_df = pickle.load(fileH)
    except:
        pass
    matrix_counter = len(param_score_df)
    if geometry_run or homology_run:
        for hh, graph_type in enumerate(conf["graph_type"]):
            for ii, patch_size in enumerate(conf["patch_size"]):
                skip_homology = False
                for jj, epsilon_factor in enumerate(conf['base_epsilon_factor']):
                    skip_geometry = False
                    if (not skip_geometry and geometry_run) or (not skip_homology and homology_run):
                        DataObj = HSI_Data(conf, patch_size, epsilon_factor, graph_type,
                                           generate_diag=False,
                                           geometry_run=geometry_run,
                                           homology_run=homology_run)
                    for kk, DiagramDistance_i in enumerate(conf["diagram_distance"]):
                        if geometry_run and not skip_geometry:
                            D = DataObj.GeometryDistanceMatrix()
                            DiagramDistance_i = 'spectral euclidean'
                        if homology_run and not skip_homology:
                            if patch_size <= 25:
                                clean_factor = conf['clean_small_homology'][str(patch_size)]
                            else:
                                clean_factor = None
                            D = DataObj.GenerateDiagDistance(DiagramDistance_i,
                                                             matrix_counter,
                                                             clean_factor)
                        if (not skip_geometry and geometry_run) or (not skip_homology and homology_run):
                            for dim in range(len(D)):
                                svm_test_set_mAP, svm_train_set_mAP, svm_val_set_mAP, svm_train_val_set_mAP, svm_config, \
                                DM_clusters_scores, DM_features_clusters, \
                                metric_cluster_summary = HSI_score.Run(D[dim],
                                                                       classification=True,
                                                                       clustering=True,
                                                                       clustering_DM=False)
                                param_score_df.loc[matrix_counter] = {'file': None,
                                                                      'graph_type': graph_type,
                                                                      'patch_size': patch_size,
                                                                      'epsilon_factor': epsilon_factor,
                                                                      'distance': DiagramDistance_i,
                                                                      'homology_dim': dim,
                                                                      'uniform_affinity_operator': conf['uniform_affinity_operator'],
                                                                      'mAP_test': svm_test_set_mAP,
                                                                      'mAP_val': svm_val_set_mAP,
                                                                      'mAP_train': svm_train_set_mAP,
                                                                      'mAP_train_val': svm_train_val_set_mAP,
                                                                      'svm_config' : svm_config,
                                                                      'DM_clusters_scores' : DM_clusters_scores,
                                                                      'DM_features_clusters' : DM_features_clusters,
                                                                      'metric_cluster_summary' : metric_cluster_summary}
                                matrix_counter += 1
                        skip_geometry = True
                        # end distance loop
                    skip_homology = True
                    # end eps loop
                if geometry_run:
                    with open('ICONES_Data/scores_geometry.pickle', 'wb') as fileH:
                        pickle.dump(param_score_df, fileH)
                if homology_run:
                    with open('ICONES_Data/scores_homology.pickle', 'wb') as fileH:
                        pickle.dump(param_score_df, fileH)

    if plot_results:
        viz.plot_full()
        viz.plot_geometry()
        viz.plot_homology()
        viz.plot_comparison()
        viz.plot_cm_tSNE()












