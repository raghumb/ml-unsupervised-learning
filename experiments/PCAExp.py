import numpy as np
from learner.PCA import PCA
from plotter import plot_curve_single, plot_pca
from clusterplotter import plot_pca_vector

class PCAExp:
    def __init__(self, reader, splitter):
        self.reader = reader
        self.splitter = splitter
        #self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.random_state = 42
        self.prefix = self.splitter.reader.dataset 


    def experiment(self):
        #self.experiment_1()
        self.experiment_best_pca_wine()

    def experiment_dimension(self):
        pca = PCA().fit(self.splitter.X_train)

        plot_curve_single(np.cumsum(pca.explained_variance_ratio_), 'PCA Variance', '# of Components', 'Variance (%)', self.prefix)


    def experiment_best(self, n_components):
        pca = PCA(n_components, random_state= self.random_state)
        data = pca.fit_transform(self.splitter.X_train)
        print('After PCA')
        print(data)
        pca_fit = pca.fit(self.splitter.X_train)
        #plot_pca(data, self.splitter.y_train, [0, 1], 'PCA Dimension Output', self.prefix)
        #plot_pca_vector(self.splitter.X_train, pca_fit)
        plot_pca(data, self.splitter.y_train,  'PCA Dimension Output', self.prefix)
        return data


    def experiment_best_test(self, n_components, dataX, dataY):
        pca = PCA(n_components, random_state= self.random_state)
        data = pca.fit_transform(dataX)
        print('After PCA')
        print(data)
        plot_pca(data, dataY, 'PCA Dimension Output', self.prefix)
        return data 




