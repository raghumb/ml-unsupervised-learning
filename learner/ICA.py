from sklearn import decomposition

class ICA:
    def __init__(self,n_components = None,
                algorithm = 'parallel',
                whiten = True,
                fun = 'logcosh',
                fun_args = None,
                max_iter = 200,
                tol = 0.0001,
                w_init = None,
                random_state = None):

        self.learner = decomposition.FastICA(n_components = n_components,
                                            algorithm = algorithm,
                                            whiten = whiten,
                                            fun = fun,
                                            fun_args = fun_args,
                                            max_iter = max_iter,
                                            tol = tol,
                                            w_init = w_init,
                                            random_state = random_state)

    def fit_transform(self, X_train):
        return self.learner.fit_transform(X_train)

    def fit(self, X_train):
        return self.learner.fit(X_train)