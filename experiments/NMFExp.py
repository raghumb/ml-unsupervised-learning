import numpy as np
from learner.NMF import NMF
from plotter import plot_curve_single

class NMFExp:
    def __init__(self, reader, splitter):
        self.reader = reader
        self.splitter = splitter
        #self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.random_state = 42
        self.prefix = self.splitter.reader.dataset 


    def experiment(self):
        self.experiment_1()
        self.experiment_best_pca_wine()

    def experiment_1(self):
        tr = NMF().fit(self.splitter.X_train, random_state = self.random_state)

        plot_curve_single(np.cumsum(tr.explained_variance_ratio_), 'NMF Variance', '# of Components', 'Variance (%)', self.prefix)


    def experiment_best_pca_wine(self):
        tr = NMF(n_components=5)
        data = tr.fit_transform(self.splitter.X_train)
        print('After NMF')
        print(data)




