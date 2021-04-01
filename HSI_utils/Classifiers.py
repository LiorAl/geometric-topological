import numpy as np
import pandas as pd
import logging
from scipy.spatial.distance import mahalanobis
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from sklearn.preprocessing import StandardScaler


class MidpointNormalize(Normalize):
    """
     Utility function to move the midpoint of a colormap to be around
    the values of interest.
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


class Classification:
    """
    Classification Class handle the classification stage by maintain statistics to each classifier request and saving
    the meta data for each classifier
    """

    ClassificationResults = []
    ClassificationResCounter = -1

    def InsertClassifierResult(self,
                               Classifiers,
                               LabelsNames,
                               EmbeddingType,
                               Lagmapp,
                               Dim=None,
                               Features=2,
                               KNeighbors=2):
        """
        insert new classifier data structure and return ID to the structure
        :param Classifiers: type of classifiers to include in the statstics
        :param LabelsNames: labels to predict in statistics
        :param Features: num of features in the embedding space
        :param Neighbors: for KNN - K
        :return: ID of the data structure
        """

        self.ClassificationResults.append({"Classification": pd.DataFrame(0.0, Classifiers, columns=LabelsNames),
                                           "EmbeddingType": EmbeddingType,
                                           "Lagmap": Lagmapp,
                                           "BetaDim": Dim,
                                           "Features": Features,
                                           "KNeighbors": KNeighbors})
        self.ClassificationResCounter += 1
        return self.ClassificationResCounter

    def CalculateLeaveOneOutACC(self, Embedding, LabelName, Label, Classifier, StatisticsID = None):

        NumOfFeatures = self.ClassificationResults[StatisticsID]["Features"]
        KNeighbors = self.ClassificationResults[StatisticsID]["KNeighbors"]

        if Classifier == 'Mahalanobis':
            Accuracy = self.NearestMahalanobisDist(Embedding, Label, NumOfFeatures)
        if Classifier == 'KNN':
            Accuracy = self.KNN(Embedding, Label, NumOfFeatures, KNeighbors)

        # Classifier result metadata, saved as key in dict classification results
        auxClassificationResults = self.ClassificationResults[StatisticsID]["Classification"]
        auxClassificationResults[LabelName][Classifier] = Accuracy


    @staticmethod
    def SVM(embedding,
            true_labels,
            **kwargs):
        """
        :param embedding: Matrix of Low dimension embedding
        KnnWeights: weight function used in prediction. Possible values:
                  ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
                  ‘distance’ : weight points by the inverse of their distance.
        :return:
        """

        visualization = kwargs.get('visualization', False)
        NumOfAttributes = true_labels.shape[0]
        predictions = np.empty(NumOfAttributes)
        best_score = 0
        best_params = {'features': None,
                       'C': None}

        loo = LeaveOneOut()
        scaler = StandardScaler()
        embedding = scaler.fit_transform(embedding)

        # Leave One Out Cross-Validation
        svm_classifier = SVC()
        param_grid = {'C': [0.01, 0.1, 1, 10, 100],
                      'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10]}
        logging.debug('Parameters Grid search: %s' % param_grid)
        clf = GridSearchCV(svm_classifier,
                           param_grid,
                           n_jobs=4,
                           cv=loo,
                           scoring='accuracy')

        clf.fit(embedding, true_labels)
        logging.debug("Best parameters set found on development set: %s" % clf.best_params_)
        logging.debug("Grid scores on development set:")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            logging.debug("%0.3f (+/-%0.03f) for %r"
                         % (mean, std * 2, params))


        if visualization:
            # visualize:
            ####################################
            # only for 2D features
            plt.figure(figsize=(8, 6))
            xx, yy = np.meshgrid(np.linspace(embedding.min()*1.1, embedding.max()*1.1, 100),
                                 np.linspace(embedding.min()*1.1, embedding.max()*1.1, 100))

            svm_classifier = SVC(C=clf.best_params_['C'], gamma=clf.best_params_['gamma'])
            svm_classifier.fit(embedding[:, 0:2], true_labels)
            Z = svm_classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            # visualize parameter's effect on decision function
            plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
            plt.scatter(embedding[:, 0], embedding[:, 1], c=true_labels, cmap=plt.cm.RdBu_r,
                        edgecolors='k')
            plt.xticks(())
            plt.yticks(())
            plt.axis('tight')

            scores = clf.cv_results_['mean_test_score'].reshape(len(param_grid['C']),
                                                                 len(param_grid['gamma']))

            plt.figure(figsize=(8, 6))
            plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
            plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
                       norm=MidpointNormalize(vmin=0.2, midpoint=0.70))
            plt.xlabel('gamma')
            plt.ylabel('C')
            plt.colorbar()
            plt.xticks(np.arange(len(param_grid['gamma'])), param_grid['gamma'], rotation=45)
            plt.yticks(np.arange(len(param_grid['C'])), param_grid['C'])
            plt.title('Validation accuracy')
            plt.show()


            # Confusion matrix for best parameters:
            svm_classifier = SVC(C=clf.best_params_['C'])
            for train_idx, test_idx in loo.split(embedding):
                X_train, X_test = embedding[train_idx, :], embedding[test_idx, :]
                y_train, y_test = true_labels[train_idx], true_labels[test_idx]
                # instance of Neighbours Classifier and fit the data
                svm_classifier.fit(X_train, y_train)
                predictions[test_idx] = svm_classifier.predict(X_test)

            Classification.ConfusionMatrix(true_labels, predictions, svm_classifier.classes_)

        return clf.best_score_, clf.best_params_


    @ staticmethod
    def NearestMahalanobisDist(Embedding, Label, NumOfFeatures):
        """
        :param matEmbedding: Matrix of Low dimension embedding
        :return:
        """
        NumOfAttributes = Label.shape[0]
        LabelClasses = np.unique(Label)
        NumOfClasses = LabelClasses.shape[0]
        CovMat = [None] * NumOfClasses
        MeanVec = [None] * NumOfClasses
        Accuracy = 0

        # Leave One Out Cross-Validation
        for LeaveInd in range(NumOfAttributes):
            LeaveMask = np.ones(NumOfAttributes)
            LeaveMask[LeaveInd] = 0
            LeaveSample = Embedding[LeaveInd, 0 : NumOfFeatures]

            # compute mean and covariance for Mahalanobis distance
            for ii, value in enumerate(LabelClasses):
                vecClassSamples = ((Label == value) * LeaveMask).astype(bool)
                CovMat[ii]  = np.cov(Embedding[vecClassSamples, 0:NumOfFeatures].T)
                MeanVec[ii] = np.mean(Embedding[vecClassSamples, 0:NumOfFeatures], axis=0)

            # classification - nearest mahalanobis distance
            ValidationDist = np.zeros(LabelClasses.shape)
            for ii in range(NumOfClasses):
                ValidationDist[ii] = mahalanobis(LeaveSample, MeanVec[ii], np.linalg.inv(CovMat[ii]))
            ValTag = LabelClasses[np.argmin(ValidationDist)]
            Accuracy += (ValTag == Label[LeaveInd])

        Accuracy /= NumOfAttributes
        return Accuracy



    @staticmethod
    def KNN(embedding,
            true_labels,
            **kwargs):
        """
        :param embedding: Matrix of Low dimension embedding
        KnnWeights: weight function used in prediction. Possible values:
                  ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
                  ‘distance’ : weight points by the inverse of their distance.
        :return:
        """
        KnnWeights = kwargs.get('KnnWeights', 'uniform')
        label_name = kwargs.get('label_name', 'no name')
        n_features_vec = kwargs.get('n_features_vec', [2, 3, 4, 5, 8, 10, 15])
        NumOfAttributes = true_labels.shape[0]
        predictions = np.empty(NumOfAttributes)
        best_score = 0
        best_params = {'features': None,
                       'n_neighbors': None,
                       'weights': None}

        loo = LeaveOneOut()
        # Leave One Out Cross-Validation
        KNNclassifier = KNeighborsClassifier()
        param_grid = {'n_neighbors': [2, 3, 4, 5, 6],
                      'weights': ['uniform', 'distance']}
        logging.info('Parameters Grid search: %s' % param_grid)
        clf = GridSearchCV(KNNclassifier,
                           param_grid,
                           n_jobs=4,
                           cv=loo,
                           scoring='f1')
        for n_features in n_features_vec:
            logging.info('KNN number of feathers: %d ' % n_features)
            clf.fit(embedding[:, 0:n_features], true_labels)
            logging.info("Best parameters set found on development set: %s" % clf.best_params_)
            logging.info("Grid scores on development set:")
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                logging.info("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            if best_score <= clf.best_score_:
                best_score = clf.best_score_
                best_params['features'] = n_features
                best_params['n_neighbors'] = clf.best_params_['n_neighbors']
                best_params['weights'] = clf.best_params_['weights']

        # Confusion matrix for best parameters:
        KNNclassifier = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], weights=best_params['weights'])
        for train_idx, test_idx in loo.split(embedding):
            X_train, X_test = embedding[train_idx, 0:best_params['features']], embedding[test_idx, 0:best_params['features']]
            y_train, y_test = true_labels[train_idx], true_labels[test_idx]
            # instance of Neighbours Classifier and fit the data
            KNNclassifier.fit(X_train, y_train)
            predictions[test_idx] = KNNclassifier.predict(X_test)

        Classification.ConfusionMatrix(true_labels, predictions, KNNclassifier.classes_)

    @staticmethod
    def LogisticRegression(embedding,
            true_labels,
            **kwargs):
        """
        :param embedding: Matrix of Low dimension embedding
        KnnWeights: weight function used in prediction. Possible values:
                  ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
                  ‘distance’ : weight points by the inverse of their distance.
        :return:
        """
        KnnWeights = kwargs.get('KnnWeights', 'uniform')
        label_name = kwargs.get('label_name', 'no name')
        n_features_vec = kwargs.get('n_features_vec', [2, 3, 4, 5, 8, 10, 15])
        NumOfAttributes = true_labels.shape[0]
        true_labels[true_labels == 0] = -1  # set labels to {-1, 1}
        predictions = np.empty(NumOfAttributes)
        best_score = 0
        best_params = {'features': None,
                       'C': None}

        loo = LeaveOneOut()
        # Leave One Out Cross-Validation
        LogistiClassifier = LogisticRegression(penalty='l1',
                                               solver='liblinear',
                                               max_iter=1000,
                                               verbose=0,
                                               tol=1e-6)
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
        logging.info('Parameters Grid search: %s' % param_grid)
        clf = GridSearchCV(LogistiClassifier,
                           param_grid,
                           n_jobs=4,
                           cv=loo,
                           scoring='f1')
        for n_features in n_features_vec:
            logging.info('Logistic Regresion number of feathers: %d ' % n_features)
            clf.fit(embedding[:, 0:n_features], true_labels)
            logging.info("Best parameters set found on development set: %s" % clf.best_params_)
            logging.info("Grid scores on development set:")
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                logging.info("%0.3f (+/-%0.03f) for %r"
                             % (mean, std * 2, params))
            if best_score <= clf.best_score_:
                best_score = clf.best_score_
                best_params['features'] = n_features
                best_params['C'] = clf.best_params_['C']

        # Confusion matrix for best parameters:
        LogistiClassifier = LogisticRegression(C=best_params['C'],
                                               penalty='l1',
                                               solver='liblinear',
                                               max_iter=10000,
                                               verbose=1,
                                               tol=1e-6)
        for train_idx, test_idx in loo.split(embedding):
            X_train, X_test = embedding[train_idx, 0:best_params['features']], embedding[test_idx,
                                                                               0:best_params['features']]
            y_train, y_test = true_labels[train_idx], true_labels[test_idx]
            # instance of Neighbours Classifier and fit the data
            LogistiClassifier.fit(X_train, y_train)
            predictions[test_idx] = LogistiClassifier.predict(X_test)

        Classification.ConfusionMatrix(true_labels, predictions, LogistiClassifier.classes_)



    @staticmethod
    def ConfusionMatrix(y_true, y_pred, labels, ymap=None, figsize=(10, 10)):
        """
        Generate matrix plot of confusion matrix with pretty annotations.
        The plot image is saved to disk.
        args:
          y_true:    true label of the data, with shape (nsamples,)
          y_pred:    prediction of the data, with shape (nsamples,)
          filename:  filename of figure file to save
          labels:    string array, name the order of class labels in the confusion matrix.
                     use `clf.classes_` if using scikit-learn models.
                     with shape (nclass,).
          ymap:      dict: any -> string, length == nclass.
                     if not None, map the labels & ys to more understandable strings.
                     Caution: original y_true, y_pred and labels must align.
          figsize:   the size of the figure plotted.
        """
        if ymap is not None:
            y_pred = [ymap[yi] for yi in y_pred]
            y_true = [ymap[yi] for yi in y_true]
            labels = [ymap[yi] for yi in labels]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_perc[i, j]
                if i == j:
                    s = cm_sum[i]
                    annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                elif c == 0:
                    annot[i, j] = ''
                else:
                    annot[i, j] = '%.1f%%\n%d' % (p, c)
        cm = pd.DataFrame(cm, index=labels, columns=labels)
        cm.index.name = 'Actual'
        cm.columns.name = 'Predicted'
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=annot, fmt='', ax=ax)
        # plt.savefig(filename)
        plt.draw()


    def PlotResults(self, LagmappList, ClassifiersList, LabelsNames, EmbeddingTypeList, KNeighbors, DimList,
                    SaveFig=False, ResultsDir=None, Title=None):

        NumOfClassifiers = len(EmbeddingTypeList)
        NumOfBarsPerTick = NumOfClassifiers * len(DimList) * len(EmbeddingTypeList)
        BarWidth = 1 / NumOfBarsPerTick
        LabelNameTicks = np.arange(len(LabelsNames))
        BarCMap = plt.get_cmap('tab10', NumOfBarsPerTick)
        BarLocation = np.linspace(-0.5 - BarWidth, 0.5 + BarWidth, NumOfBarsPerTick)
        # LagmappList = [LagmappList] # TODO: fix list of Lagmapps
        NumOfSubplots = len(LagmappList)

        fig, axes = plt.subplots(1, NumOfSubplots, figsize=(19.20, 10.80))
        if not isinstance(axes, np.ndarray):
            # for indexing purpose
            axes = [axes]

        for ii, RequireLagmapp in enumerate(LagmappList):
            BarCounter = 0
            for ClassiResults in self.ClassificationResults:
                Results = ClassiResults["Classification"]
                EmbeddingType = ClassiResults["EmbeddingType"]
                Lagmapp = ClassiResults["Lagmap"]
                BetaDim = ClassiResults["BetaDim"]
                Features = ClassiResults["Features"]
                KNeighbors = ClassiResults["KNeighbors"]
                Legend = []

                if RequireLagmapp == Lagmapp:
                    fig.suptitle("Classisification Results - " + Title)
                    for Classifier in ClassifiersList:
                        rects1 = axes[ii].bar(LabelNameTicks - 0.5 * BarLocation[BarCounter],
                                              Results.loc[Classifier],
                                              width=BarWidth, color=BarCMap(BarCounter),
                                              label=r"{} $\beta_{}$ {}".format(EmbeddingType, BetaDim, Classifier))
                        BarCounter += 1
                        # Add some text for labels, title and custom x-axesis tick labels, etc.
            # end ClassificationResults loop

            axes[ii].set_ylabel('Accuracy')
            axes[ii].set_title('{} Features, {}-NN'.format(Features, KNeighbors))
            axes[ii].set_xticks(LabelNameTicks)
            axes[ii].set_xticklabels(LabelsNames)
            axes[ii].set_yticks(np.arange(0, 1, 0.05))
            axes[ii].legend()
            axes[ii].yaxis.grid()
            axes[ii].set_ylim(0, 1)
            axes[ii].set_title("Lagmap:  {}".format(Lagmapp))

        if SaveFig:
            fig.savefig(ResultsDir + '/Classification' +  Title + ".jpg")



