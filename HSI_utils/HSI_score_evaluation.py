import numpy as np
import logging
import pandas as pd
import DimensionReduction as DR
from sklearn.metrics import auc, accuracy_score, confusion_matrix, average_precision_score, make_scorer
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.manifold import TSNE, MDS
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
import plotly.express as px


class evaluation:
    DM_features = [2, 5, 10, 15, 20, 25, 30, 35, 40, 50]
    DM_eps_factor = [0.1, 0.5, 1, 2, 5]
    n_classification_features = 20
    affinity_uniformity = [True, False]
    best_score = -1
    best_params = None
    best_eps = None
    best_n = None
    params = None
    DM_embedding = None

    PH_dimension = 2
    kfold_n_splits = 10

    class_names = ['Agriculture', 'Cloud', 'Desert', 'Dense_urban',
                    'Forest', 'Mountain', 'Ocean', 'Snow', 'Wetland']

    def __init__(self, labels):
        self.labels = labels
        self.n_labels = self.labels.shape[0]
        self.n_classes = np.max(self.labels)
        self.one_hot_labels = label_binarize(self.labels, classes=np.unique(self.labels))
        self.n_labels_in_class = np.sum(self.one_hot_labels, axis=0)
        self.class_weight = self.n_labels_in_class / self.n_labels
        self.svm_class_wieght = {ii: value for ii, value in enumerate(self.class_weight)}
        self.scaler = StandardScaler()

        self.skf_val = StratifiedKFold(n_splits=self.kfold_n_splits, shuffle=True, random_state=0)
        self.skf_test = StratifiedKFold(n_splits=self.kfold_n_splits, shuffle=True, random_state=42)

    def Run(self, D, classification=True, clustering=True, clustering_DM=True):

        svm_test_set_mAP = np.zeros((len(self.DM_eps_factor), len(self.affinity_uniformity), 2)) # dim -1 mean and std
        svm_train_set_mAP = np.zeros((len(self.DM_eps_factor), len(self.affinity_uniformity), 2))
        svm_val_set_mAP = np.zeros((len(self.DM_eps_factor), len(self.affinity_uniformity), 2))
        svm_train_val_set_mAP = np.zeros((len(self.DM_eps_factor), len(self.affinity_uniformity), 2))
        svm_config = []
        DM_clusters_scores = np.zeros((len(self.DM_eps_factor), len(self.affinity_uniformity), 3))  # 3 for CH, DB anf SI
        DM_features_clusters = []

        # Diffusion maps decomposition
        for eps_idx, eps_factor in enumerate(self.DM_eps_factor):
            for aff_idx, affinity_uniformity_flag in enumerate(self.affinity_uniformity):
                DM_embedding = DR.DiffusionMaps(D,
                                                epsilon_factor=eps_factor,
                                                uniformity=affinity_uniformity_flag)
                if DM_embedding['complex'] or np.any(np.isnan(DM_embedding['embedding'])) :
                    continue

                # scaling the features
                embedding = DM_embedding['embedding']
                embedding = self.scaler.fit_transform(embedding)
                # Classification
                if classification:
                    clf, train_score, test_score, val_score, train_val_score = self.classification(embedding)
                    svm_test_set_mAP[eps_idx, aff_idx, :] = test_score
                    svm_train_set_mAP[eps_idx, aff_idx, :] = train_score
                    svm_val_set_mAP[eps_idx, aff_idx, :] = val_score
                    svm_train_val_set_mAP[eps_idx, aff_idx, :] = train_val_score
                    svm_config.append(clf)

                if clustering_DM:
                    dm_cluster_summary = self.diffusion_maps_clusters(embedding)
                    DM_clusters_scores[eps_idx, aff_idx, :] = [dm_cluster_summary['calinski_harabasz'].max(),
                                                               dm_cluster_summary['davies_bouldin'].max(),
                                                               dm_cluster_summary['silhouette'].max()]
                    DM_features_clusters.append(dm_cluster_summary)

        if clustering:
            metric_cluster_summary = self.metric_clusters(D)

        logging.info("SVM: | Test: %.2f | Train:  %.2f | Validation: %.2f | Train-Val %.2f |"
                     % (svm_test_set_mAP.max(),
                        svm_train_set_mAP.max(),
                        svm_val_set_mAP.max(),
                        svm_train_val_set_mAP.max()))
        logging.info("DM clusters: | Calinski-Harabasz: %.2f | Davies-Bouldin:  %.2f  | Silhouette: %.2f| "
                     % (DM_clusters_scores[:, :, 0].max(),
                        DM_clusters_scores[:, :, 1].max(),
                        DM_clusters_scores[:, :, 2].max()))
        logging.info("Clusters: | %s |" % metric_cluster_summary)

        return svm_test_set_mAP, svm_train_set_mAP, svm_val_set_mAP, svm_train_val_set_mAP, svm_config, \
               DM_clusters_scores, DM_features_clusters,\
               metric_cluster_summary


    def classification(self, embedding):
        svm_param_grid = {'C': [0.1, 1, 10, 100, 1000],
                          'gamma': ['scale', 'auto']}

        clf_list = []
        mAP_test = np.zeros(self.kfold_n_splits)
        mAP_train = np.zeros(self.kfold_n_splits)

        mAP_val = np.zeros((2, self.kfold_n_splits))  # 1st row mean and 2nd row std
        mAP_train_val = np.zeros((2, self.kfold_n_splits)) # train set withpit the validation i.e. 60%


        scorer = make_scorer(lambda y_true, y_prob :
                             average_precision_score(label_binarize(y_true, classes=np.unique(self.labels)),
                                                     y_prob,
                                                     average='micro'),
                             needs_proba=True)
        multiclass_svm = SVC(decision_function_shape='ovr',
                             probability=True,
                             class_weight=self.svm_class_wieght)
        clf = GridSearchCV(multiclass_svm,
                           svm_param_grid,
                           scoring=scorer,
                           cv=self.skf_val,
                           return_train_score=True)
        idx = 0
        for train_index, test_index in self.skf_test.split(embedding, self.labels):
            clf.fit(embedding[train_index, 0:self.n_classification_features], self.labels[train_index])
            clf_list.append(clf.cv_results_)
            mAP_val[:, idx] = [clf.cv_results_['mean_test_score'][clf.best_index_],
                               clf.cv_results_['std_test_score'][clf.best_index_]]
            mAP_train_val[:, idx] = [clf.cv_results_['mean_train_score'][clf.best_index_],
                                     clf.cv_results_['std_train_score'][clf.best_index_]]
            clf.best_estimator_.fit(embedding[train_index, 0:self.n_classification_features], self.labels[train_index])
            train_prob = clf.best_estimator_.predict_proba(embedding[train_index, 0:self.n_classification_features])
            test_prob = clf.best_estimator_.predict_proba(embedding[test_index, 0:self.n_classification_features])
            mAP_train[idx] = average_precision_score(label_binarize(self.labels[train_index], classes=np.unique(self.labels)),
                                                      train_prob, average='micro')
            mAP_test[idx] = average_precision_score(label_binarize(self.labels[test_index], classes=np.unique(self.labels)),
                                                    test_prob, average='micro')
            idx += 1

        train_score     = [np.mean(mAP_train), np.std(mAP_train)]
        test_score      = [np.mean(mAP_test), np.std(mAP_test)]
        val_score       = np.mean(mAP_val, axis=1)
        train_val_score = np.mean(mAP_train_val, axis=1)

        return clf_list, train_score, test_score, val_score, train_val_score


    def diffusion_maps_clusters(self, embedding):

        CH = np.zeros(len(self.DM_features))
        DB = np.zeros(len(self.DM_features))
        SI = np.zeros(len(self.DM_features))

        for idx, n_features in enumerate(self.DM_features):
            CH[idx] = calinski_harabasz_score(embedding[:, 0:n_features], self.labels) # better high
            DB[idx] = davies_bouldin_score(embedding[:, 0:n_features], self.labels) # better high
            SI[idx] = silhouette_score(embedding[:, 0:n_features], self.labels) # -1 for incorrect

        return pd.DataFrame({"calinski_harabasz": CH,
                "davies_bouldin": DB,
                "silhouette": SI,
                "features": self.DM_features})

    def metric_clusters(self, D):
        neigh = NearestNeighbors(n_neighbors=9, metric='precomputed')
        neigh.fit(D)
        _, neighbors = neigh.kneighbors()
        cluster_score = np.zeros(5)
        for idx, n_neighbors in enumerate(range(1, 11, 2)):
            neighbors_aux = neighbors[:, 0:n_neighbors]
            neighbors_labels = self.labels[neighbors_aux.flatten()].reshape(neighbors_aux.shape)
            labels = np.tile(np.expand_dims(self.labels, axis=1), (1, n_neighbors))
            true_category = np.sum(neighbors_labels == labels, axis=1) > n_neighbors//2
            cluster_score[idx] = np.sum(true_category) / self.labels.shape[0]
        return cluster_score


    def tSNE_clusters(self, D):
        perplexity = [5, 10, 15, 20, 30, 50]
        n_iter = 2000

        best_score = -1
        best_params = None

        for p in perplexity:
            tSNE_embedding = TSNE(n_components=2,
                                  perplexity=p,
                                  n_iter=n_iter,
                                  metric='precomputed').fit_transform(D)
            tSNE_score = {"calinski_harabasz" : calinski_harabasz_score(tSNE_embedding, self.labels), # better high
                          "davies_bouldin"    :  davies_bouldin_score(tSNE_embedding, self.labels), # better high
                          "silhouette"        : silhouette_score(tSNE_embedding, self.labels)} # -1 for incorrect

            if best_score < tSNE_score["calinski_harabasz"]:
                best_score = tSNE_score["calinski_harabasz"]
                best_params = {'score': tSNE_score,
                               'p': p}

        return {'tSNE': best_score,
                'tSNE_params': best_params}



    @staticmethod
    def classifier_score(labels, one_hot_labels, labels_pred, labels_prob):
        AP = np.zeros(np.max(labels) + 1)
        top_20 = np.zeros(np.max(labels) + 1)
        top_10 = np.zeros(np.max(labels) + 1)
        for idx in range(np.max(labels) + 1):
            AP[idx] = average_precision_score(one_hot_labels[:, idx], labels_prob[:, idx])
            sort_idx = np.argsort(labels_prob[:, idx], axis=0)[::-1]
            top_20[idx] = np.sum(labels_pred[sort_idx[0:20]] == idx)/20
            top_10[idx] = np.sum(labels_pred[sort_idx[0:10]] == idx)/10
        cf = confusion_matrix(labels, labels_pred)
        mAP = average_precision_score(one_hot_labels, labels_prob, average="micro")

        return {'mAP' : mAP,
                'AP'  : AP,
                'top_20' : top_20,
                'top_10' : top_10,
                'confusion_matrix' : cf}


    def plot_best_results(self, D):

        DM_vec_idx = np.arange(0, 6)
        perplexity = [25, 35, 45, 55]
        early_exaggeration_vec = [4, 8, 12, 16, 20, 24]
        lr = [150, 200, 250]
        n_iter = 2000
        DM_eps_factor = [0.5, 1, 2]

        # clustering


        for p in perplexity:
            for exaggeration in early_exaggeration_vec:
                for learning_rate in lr:
                    embedding = TSNE(n_components=2,
                                     perplexity=p,
                                     n_iter=n_iter,
                                     early_exaggeration=exaggeration,
                                     learning_rate=learning_rate,
                                     metric='precomputed',
                                     verbose=0).fit_transform(D)
                    print('p: %d | exa: %d | lr:  %d' %(p, exaggeration, learning_rate))
                    print("Calinski Harabasz %.2f" % calinski_harabasz_score(embedding, self.labels))  # better high
                    print("Davies Bouldin %.2f" % davies_bouldin_score(embedding, self.labels)) #  better high
                    print("Silhouette %.2f" % silhouette_score(embedding, self.labels)) # 1 for incorrect clustering and +1 for highly dense clustering. Scores around zero indicate overlapping clusters.

        embedding = TSNE(n_components=2,
                         perplexity=45,
                         n_iter=n_iter,
                         early_exaggeration=12,
                         learning_rate=150,
                         metric='precomputed',
                         verbose=0).fit_transform(D)


        MetricMDS = MDS(n_components=2, max_iter=1000, eps=1e-6,
                        dissimilarity="precomputed").fit(D).embedding_
        print("Calinski Harabasz %.2f" % calinski_harabasz_score(MetricMDS, self.labels))  # better high
        print("Davies Bouldin %.2f" % davies_bouldin_score(MetricMDS, self.labels))  # better high
        print("Silhouette %.2f" % silhouette_score(MetricMDS, self.labels))  # 1 for incorrect
        NonMetricMDS = MDS(n_components=2, metric=False, max_iter=1000, eps=1e-6,
                               dissimilarity="precomputed", n_init=1).fit_transform(D, init=MetricMDS)
        print("Calinski Harabasz %.2f" % calinski_harabasz_score(NonMetricMDS, self.labels))  # better high
        print("Davies Bouldin %.2f" % davies_bouldin_score(NonMetricMDS, self.labels))  # better high
        print("Silhouette %.2f" % silhouette_score(NonMetricMDS, self.labels))  # 1 for incorrect
        embedding = MetricMDS

        colormap = px.colors.qualitative.Set1
        raw_symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'octagon', 'star', 'hourglass']
        # tSNE:

        fig = go.Figure()
        # plot Responder label
        for idx, label in enumerate(self.class_names):
            fig.add_trace(go.Scatter(x=embedding[self.one_hot_labels[:, idx] == 1, 0], y=embedding[self.one_hot_labels[:,idx] ==1, 1],
                                     marker_symbol=raw_symbols[idx],
                                     mode='markers',
                                     marker=dict(color=idx,
                                                 size=12,
                                                 colorscale=colormap),
                                     showlegend=True,
                                     name=label,
                                     text=self.labels))
        fig.show()

