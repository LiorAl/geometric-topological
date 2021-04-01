import os
import glob
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from skimage.transform import resize
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.io as pio
import spectral.io.envi as envi
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, average_precision_score
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.neighbors import NearestNeighbors
import pickle
import plotly.figure_factory as ff
import DimensionReduction as DR

pio.renderers.default = "browser"

def PCA_SVM():
    data_dir = os.getcwd() + '/ICONES_Data/ICONES-HSI/'
    labels_names = ['Agriculture', 'Cloud', 'Desert', 'Dense_urban',
                    'Forest', 'Mountain', 'Ocean', 'Snow', 'Wetland']
    n_features = 20
    labels = []
    H, W = 150, 150
    n_channels = 224
    n_samples = 486
    n_bands = 5
    kfold_n_splits = 10
    data = np.zeros((n_samples, W * H * n_bands))
    pca_spectral_band = PCA(n_components=1)
    skf = StratifiedKFold(n_splits=kfold_n_splits, shuffle=True, random_state=0)

    bands_index = np.linspace(0, n_channels, n_bands+1, dtype=int)
    data_counter = 0
    with tqdm(total=n_samples) as pbar:
        for label_idx, label_dir in enumerate(labels_names):
            img_files = glob.glob('%s/%s/*.img' % (data_dir, label_dir))
            hdr_files = glob.glob('%s/%s/*.hdr' % (data_dir, label_dir))
            for file in zip(hdr_files, img_files):
                labels.append(label_idx)
                img_header = envi.open(file[0], file[1])
                img = np.array(img_header.open_memmap(writeable=True))
                resized_img = resize(img, (H, W, n_channels)).reshape(H * W, n_channels)
                spectral_pca_embedding = [pca_spectral_band.fit_transform(resized_img[:, bands_index[ii]: bands_index[ii + 1]])
                                          for ii in range(n_bands)]
                spectral_pca_embedding = np.squeeze(np.array(spectral_pca_embedding).T).flatten()
                data[data_counter, :] = spectral_pca_embedding
                data_counter += 1
                pbar.update(1)

    with open('ICONES_Data/ICONES-HSI/labels.pickle', 'rb') as file:
        labels = pickle.load(file)


    D = squareform(pdist(data))
    labels = np.array(labels)

    DM_embedding = DR.DiffusionMaps(D,
                                    epsilon_factor=1,
                                    uniformity=True)

    # scaling the features
    embedding = DM_embedding['embedding']
    scaler = StandardScaler()
    embedding = scaler.fit_transform(embedding[:, 0:n_features])

    # classifcation
    one_hot_labels = label_binarize(labels, classes=np.unique(labels))

    # SVM
    n_labels_in_class = np.sum(one_hot_labels, axis=0)
    class_weight = n_labels_in_class / n_samples
    class_weight = {ii: value for ii, value in enumerate(class_weight)}

    skf_val = StratifiedKFold(n_splits=kfold_n_splits, shuffle=True, random_state=0)
    skf_test = StratifiedKFold(n_splits=kfold_n_splits, shuffle=True, random_state=42)

    svm_param_grid = {'C': [0.1, 1, 10, 100, 1000],
                      'gamma': ['scale', 'auto', 1, 10]}


    clf_list = []
    mAP_test = np.zeros(kfold_n_splits)
    mAP_train = np.zeros(kfold_n_splits)

    mAP_val = np.zeros((2, kfold_n_splits))  # 1st row mean and 2nd row std
    mAP_train_val = np.zeros((2, kfold_n_splits))  # train set withpit the validation i.e. 60%

    scorer = make_scorer(lambda y_true, y_prob:
                         average_precision_score(label_binarize(y_true, classes=np.unique(labels)),
                                                 y_prob,
                                                 average='micro'),
                         needs_proba=True)
    multiclass_svm = SVC(decision_function_shape='ovr',
                         probability=True,
                         class_weight=class_weight)
    clf = GridSearchCV(multiclass_svm,
                       svm_param_grid,
                       scoring=scorer,
                       cv=skf_val,
                       return_train_score=True)
    idx = 0
    for train_index, test_index in skf_test.split(embedding, labels):
        clf.fit(embedding[train_index, 0: n_features], labels[train_index])
        clf_list.append(clf.cv_results_)
        mAP_val[:, idx] = [clf.cv_results_['mean_test_score'][clf.best_index_],
                           clf.cv_results_['std_test_score'][clf.best_index_]]
        mAP_train_val[:, idx] = [clf.cv_results_['mean_train_score'][clf.best_index_],
                                 clf.cv_results_['std_train_score'][clf.best_index_]]
        clf.best_estimator_.fit(embedding[train_index, 0:n_features], labels[train_index])
        train_prob = clf.best_estimator_.predict_proba(embedding[train_index, 0:n_features])
        test_prob = clf.best_estimator_.predict_proba(embedding[test_index, 0:n_features])
        mAP_train[idx] = average_precision_score(
            label_binarize(labels[train_index], classes=np.unique(labels)),
            train_prob, average='micro')
        mAP_test[idx] = average_precision_score(label_binarize(labels[test_index], classes=np.unique(labels)),
                                                test_prob, average='micro')
        idx += 1

    train_score = [np.mean(mAP_train), np.std(mAP_train)]
    test_score = [np.mean(mAP_test), np.std(mAP_test)]
    val_score = np.mean(mAP_val, axis=1)
    train_val_score = np.mean(mAP_train_val, axis=1)

    # #### kNN clusters #####
    neigh = NearestNeighbors(n_neighbors=9, metric='precomputed')
    neigh.fit(D)
    _, neighbors = neigh.kneighbors()
    cluster_score = np.zeros(5)
    for idx, n_neighbors in enumerate(range(1, 11, 2)):
        neighbors_aux = neighbors[:, 0:n_neighbors]
        neighbors_labels = labels[neighbors_aux.flatten()].reshape(neighbors_aux.shape)
        labels_aux = np.tile(np.expand_dims(labels, axis=1), (1, n_neighbors))
        true_category = np.sum(neighbors_labels == labels_aux, axis=1) > n_neighbors // 2
        cluster_score[idx] = np.sum(true_category) / labels_aux.shape[0]


    print("Train mAP: %.3f +- %.3f| Test mAP: %.3f +- %.3f, Val mAP: %.3f +- %.3f, Train_Val mAP: %.3f +- %.3f "
          % (train_score[0], train_score[1],
             test_score[0], test_score[1],
             val_score[0], val_score[1],
             train_val_score[0], train_val_score[1],))
    print("kNN clusters: %.3f" % cluster_score.max())

    train_cm = np.zeros((np.unique(labels).shape[0], np.unique(labels).shape[0], kfold_n_splits))
    test_cm = np.zeros((np.unique(labels).shape[0], np.unique(labels).shape[0], kfold_n_splits))

    best_classifier = clf.best_estimator_
    # train and test confusion matrix
    iter = 0
    for train_index, test_index in skf.split(embedding, labels):
        best_classifier.fit(embedding[train_index, :], labels[train_index])
        train_pred = best_classifier.predict(embedding[train_index, :])
        test_pred = best_classifier.predict(embedding[test_index, :])
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
    z_text = [["%.2f +- %.2f" %(m, s) for m, s in zip(mm, ss)] for mm, ss in zip(mean_train_cm, std_train_cm)]
    fig = ff.create_annotated_heatmap(mean_train_cm, x=x, y=y, annotation_text=z_text, colorscale='Viridis')
    fig.update_layout(title_text='<i><b>Train Confusion matrix</b></i>')
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=-0.35,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    fig.update_layout(
        xaxis_title="Predicted value",
        yaxis_title="Real value",
        xaxis=dict(autorange='reversed'))

    # add colorbar
    fig['data'][0]['showscale'] = True
    fig.show()

    z_text = [["%.2f +- %.2f" % (m, s) for m, s in zip(mm, ss)] for mm, ss in zip(mean_test_cm, std_test_cm)]
    fig = ff.create_annotated_heatmap(mean_test_cm, x=x, y=y, annotation_text=z_text, colorscale='Viridis')
    fig.update_layout(title_text='<i><b>Test Confusion matrix</b></i>',
                      # xaxis = dict(title='x'),
                      # yaxis = dict(title='x')
                      )
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=-0.35,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))
    fig.update_layout(
        xaxis_title="Predicted value",
        yaxis_title="Real value",
        xaxis=dict(autorange='reversed'))
    fig['data'][0]['showscale'] = True
    fig.show()











