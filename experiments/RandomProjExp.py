import numpy as np
from learner.RandomProjection import RandomProjection
from plotter import plot_curve_single,plot_pca


class RandomProjExp:
    def __init__(self, reader, splitter):
        self.reader = reader
        self.splitter = splitter
        #self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.random_state = 42
        self.prefix = self.splitter.reader.dataset 


    def experiment(self):
        self.experiment_1()
        self.experiment_best_pca_wine()

    def experiment_cluster_size(self, dataX, prefix):
        K = range(3,15)
        X = dataX
        min_error = 1000.
        min_k = 0
        for k in K:        
            rp = RandomProjection(n_components = k, random_state = self.random_state)
            data = rp.fit_transform(dataX)
        
            print('shape of RP')
            print(data.shape)
            inverse_data = rp.inverse_transform(data)
            error = ((dataX - inverse_data) ** 2).mean()
            print('error ')
            print(error)
            if error < min_error:
                min_error = error
                min_k = k

            
        print('min k is ' + str(k))
        #plot_pca(data, self.splitter.y_train, [0, 1], 'Random Proj Dimension Output', 'wine')
        


    def experiment_best(self, n_components):
        tr = RandomProjection(n_components = n_components, random_state = self.random_state)
        data = tr.fit_transform(self.splitter.X_train)
        print('shape of RP')
        print(data.shape)
        return data

    def experiment_best_test(self, n_components, dataX, dataY):
        tr = RandomProjection(n_components = n_components, random_state = self.random_state)
        data = tr.fit_transform(dataX)
        print('shape of RP')
        print(data.shape)
        return data        


