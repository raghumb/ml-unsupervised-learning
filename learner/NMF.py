from sklearn import decomposition
import numpy as np

class NMF:
    def __init__(self,
                n_components = None,
                init = None,
                solver = 'cd',
                beta_loss = 'frobenius',
                tol = 0.0001,
                max_iter = 200,
                random_state = None,
                alpha = 0.0,
                l1_ratio = 0.0,
                verbose = 0,
                shuffle = False):
        
        self.learner = decomposition.NMF(                
                n_components = n_components,
                init = init,
                solver = solver,
                beta_loss = beta_loss,
                tol = tol,
                max_iter = max_iter,
                random_state = random_state,
                alpha = alpha,
                l1_ratio = l1_ratio,
                verbose = verbose,
                shuffle = shuffle)


    def fit_transform(self, X_train):
        return self.learner.fit_transform(X_train)
        

    def fit(self, X_train):
        return self.learner.fit(X_train)

