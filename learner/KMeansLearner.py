from sklearn.cluster import KMeans
import time

class KMeansLearner:
    def __init__(self,
                n_clusters=8, 
                init = 'k-means++', 
                n_init = 10,
                max_iter = 300,
                tol = 0.0001,
                precompute_distances = 'auto',
                verbose = 0,
                random_state = None,
                copy_x = True,
                n_jobs = None,
                algorithm = 'auto'):
    

        self.learner = KMeans(n_clusters=n_clusters,
                            init = 'k-means++', 
                            n_init = n_init,
                            max_iter = max_iter,
                            tol = tol,
                            precompute_distances = precompute_distances,
                            verbose = verbose,
                            random_state = random_state,
                            copy_x = copy_x,
                            n_jobs = n_jobs,
                            algorithm = algorithm)


    def fit(self, X_train):
        k_fit = self.learner.fit(X_train)
        print(k_fit.labels_)
        print(k_fit.cluster_centers_)
        return k_fit

    def fit_predict(self, X_train):
        return self.learner.fit_predict(X_train)


    def predict(self, y_test):
        #TODO
        pass




    