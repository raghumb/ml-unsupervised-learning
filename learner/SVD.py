from sklearn import decomposition
import numpy as np

class SVD:
    def __init__(self,
                n_components = None,
                algorithm = 'randomized',
                n_iter = 5,
                tol = 0.0,
                random_state = None):
        
        self.learner = decomposition.TruncatedSVD(n_components = n_components,
                        algorithm = algorithm,
                        n_iter = n_iter,
                        tol = tol,
                        random_state = random_state)


    def fit_transform(self, X_train):
        return self.learner.fit_transform(X_train)
        

    def fit(self, X_train):
        return self.learner.fit(X_train)

