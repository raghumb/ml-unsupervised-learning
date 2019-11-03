from sklearn.mixture import GaussianMixture
import time

class EMLearner:
    def __init__(self,
            n_components = 1,
            covariance_type = 'full',
            tol = 0.001,
            reg_covar = 1e-06,
            max_iter = 100,
            n_init = 1,
            init_params = 'kmeans',
            weights_init = None,
            means_init = None,
            precisions_init = None,
            random_state = None,
            warm_start = False,
            verbose = 0,
            verbose_interval = 10):
    

        self.learner = GaussianMixture(n_components = n_components,
                            covariance_type = covariance_type,
                            tol = tol,
                            reg_covar = reg_covar,
                            max_iter = max_iter,
                            n_init = n_init,
                            init_params = init_params,
                            weights_init = weights_init,
                            means_init = means_init,
                            precisions_init = precisions_init,
                            random_state = random_state,
                            warm_start = warm_start,
                            verbose = verbose,
                            verbose_interval = verbose_interval)


    def fit(self, X_train):
        k_fit = self.learner.fit(X_train)
        #print(k_fit.labels_)
        #print(k_fit.cluster_centers_)
        return k_fit

    def fit_predict(self, X_train):
        return self.learner.fit_predict(X_train)


    def predict(self, y_test):
        #TODO
        pass




    