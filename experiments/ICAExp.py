import numpy as np
from learner.ICA import ICA
from plotter import plot_curve_single, plot_kurtosis
import pandas as pd

class ICAExp:
    def __init__(self, reader, splitter):
        self.reader = reader
        self.splitter = splitter
        #self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.random_state = 42
        self.prefix = self.splitter.reader.dataset 
        self.rand_seeds = self.create_random_seeds()


    def create_random_seeds(self):
        seed = range(3,15)
        return np.array(seed)
    
    def experiment(self):
        #self.experiment_1()
        self.experiment_best_ica_wine()

    def experiment_dimension(self, dataX):
        K = range(3,15)
        kurt_max = 0.
        k_opt = 0
        X = dataX
        for i in range(len(self.rand_seeds)):
            rand_state = self.rand_seeds[i]        
            k = rand_state
            data = ICA(random_state = rand_state).fit_transform(X)
            #print('ICA')
            #print(data)
            df =  pd.DataFrame(data = data)
            kurt = df.kurt(axis = 0) 
            #print('kurt')
            #print(kurt)
            avgKurt = np.mean(kurt)
            print('avgKurt')
            print(avgKurt)
            print('k ' + str(k))
            if avgKurt > kurt_max:
                kurt_max = avgKurt
                k_opt = k

        print('optimal k with highest Kurtosis is ' + str(k_opt))

        #plot_curve_single(np.cumsum(pca.explained_variance_ratio_), 'ICA Variance', '# of Components', 'Variance (%)', self.prefix)


    def experiment_best(self, n_components):
        pca = ICA(n_components = n_components)
        data = pca.fit_transform(self.splitter.X_train)
        print('After ICA')
        print(data)
        df =  pd.DataFrame(data = data)
        kurt = df.kurt(axis = 0) 
        plot_kurtosis(kurt)
        return data

    def experiment_best_test(self, n_components, dataX, dataY):
        pca = ICA(n_components = n_components)
        data = pca.fit_transform(dataX)
        print('After ICA')
        print(data)
        return data        




