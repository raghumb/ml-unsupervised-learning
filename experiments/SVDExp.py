import numpy as np
from learner.SVD import SVD
from plotter import plot_curve_single, plot_pca

class SVDExp:
    def __init__(self, reader, splitter):
        self.reader = reader
        self.splitter = splitter
        #self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.random_state = 42
        self.prefix = self.splitter.reader.dataset 


    def experiment(self):
        #self.experiment_1()
        self.experiment_best_pca_wine()

    def experiment_dimension(self, dataX, prefix):
        n_components = self.splitter.X_train.shape[1] - 1
        print('n_components '+ str(n_components))
        svd = SVD(n_components = n_components).fit(dataX)

        plot_curve_single(np.cumsum(svd.explained_variance_ratio_), 'SVD Variance', '# of Components', 'Variance (%)', self.prefix)


    def experiment_best(self, dataX, n_components):
        svd = SVD(n_components = n_components, random_state= self.random_state)
        data = svd.fit_transform(dataX)
        print('After SVD')
        print(data)
        plot_pca(data, self.splitter.y_train,  'SVD Dimension Output', self.prefix)
        return data

    def experiment_best_test(self, n_components, dataX, dataY): 
        svd = SVD(n_components = n_components, random_state= self.random_state)
        data = svd.fit_transform(dataX)
        print('After SVD')
        print(data)
        plot_pca(data, dataY, [0, 1], 'SVD Dimension Output', self.prefix)
        return data               





