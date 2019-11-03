from sklearn import decomposition
import numpy as np

class PCA:
    def __init__(self,
                n_components = None,
                copy = True,
                whiten = False,
                svd_solver = 'auto',
                tol = 0.0,
                iterated_power = 'auto',
                random_state = None):
        
        self.learner = decomposition.PCA(n_components = n_components,
                        copy = copy,
                        whiten = whiten,
                        svd_solver = svd_solver,
                        tol = tol,
                        iterated_power = iterated_power,
                        random_state = random_state)


    def fit_transform(self, X_train):
        return self.learner.fit_transform(X_train)
        

    def fit(self, X_train):
        return self.learner.fit(X_train)

