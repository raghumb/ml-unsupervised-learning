from sklearn import random_projection
import numpy as np

class RandomProjection:
    def __init__(self,
                n_components = 'auto',
                eps = 0.1,
                random_state = None
                ):
        
        self.learner = random_projection.GaussianRandomProjection(                
                n_components = n_components,
                eps = eps,
                random_state = random_state
                )


    def inverse_transform(self, X):
        return np.dot(X, self.learner.components_)
    
    def fit_transform(self, X_train):
        return self.learner.fit_transform(X_train)
        

    def fit(self, X_train):
        return self.learner.fit(X_train)

