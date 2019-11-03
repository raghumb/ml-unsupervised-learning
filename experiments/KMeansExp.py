from learner.KMeansLearner import KMeansLearner
from experiments.RandomProjExp import RandomProjExp
from experiments.PCAExp import PCAExp
from experiments.ICAExp import ICAExp
from experiments.SVDExp import SVDExp
from plotter import plot_curve
#from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_samples, silhouette_score,davies_bouldin_score
import numpy as np
from scipy.spatial.distance import cdist
from clusterplotter import plot_cluster, plot_cluster_wo_centroid
from sklearn.metrics import homogeneity_completeness_v_measure, adjusted_rand_score

class KMeansExp:
    def __init__(self, reader, splitter, run_final):
        self.reader = reader
        self.splitter = splitter
        #self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.random_state = 42
        self.run_final = run_final
        #self.splitter.read_split_data()
        
    def experiment(self):
        print('dataset '+ str(self.splitter.reader.dataset))
        if self.splitter.reader.dataset == 'wine':
            print('wine flow')
            self.experiment_wine()
            
        else:            
            print('bank flow')
            self.experiment_bank()
    
    
    def experiment_wine(self):

        val = 4
        test = False
        
        if self.eval_condition(val, 0, test) == True:
            # without Dim reduction
            prefix = 'No Dim Red'
            n_components = 5   
            if self.run_final == False:
                self.experiment_cluster_size(self.splitter.X_train, prefix)
            else:
                self.experiment_cluster(self.splitter.X_train, self.splitter.y_train, prefix, n_components)


        #With PCA
        if self.eval_condition(val, 1, test) == True:
            prefix = 'PCA'
            er = PCAExp(self.reader, self.splitter)
            er.experiment_dimension()
            n_components = 7        
            dataX = er.experiment_best(n_components)   
            dataX_combined = dataX #np.concatenate((self.splitter.X_train, dataX), axis = 1)   
            if self.run_final == False:  
                self.experiment_cluster_size(dataX_combined, prefix)
            else:
                self.experiment_cluster(dataX_combined, self.splitter.y_train, prefix, n_components)


         #With ICA
        if self.eval_condition(val, 2, test) == True:
            prefix = 'ICA'
            er = ICAExp(self.reader, self.splitter)
            er.experiment_dimension(self.splitter.X_train)
            n_components = 11      
            dataX = er.experiment_best(n_components)   
            dataX_combined = dataX #np.concatenate((self.splitter.X_train, dataX), axis = 1)   
            
            if self.run_final == False:  
                self.experiment_cluster_size(dataX_combined, prefix)
            else:
                n_components = 12
                self.experiment_cluster(dataX_combined, self.splitter.y_train, prefix, n_components)
            

        #With Random Proj
        if self.eval_condition(val, 3, test) == True:
            prefix = 'RP'            
            er = RandomProjExp(self.reader, self.splitter)
            er.experiment_cluster_size(self.splitter.X_train, prefix)       
            n_components = 14
            dataX = er.experiment_best(n_components) 
            dataX_combined = dataX
            if self.run_final == False:  
                self.experiment_cluster_size(dataX_combined, prefix)
            else:
                n_components = 6
                self.experiment_cluster(dataX_combined, self.splitter.y_train, prefix, 5)

        #With SVD
        if self.eval_condition(val, 4, test) == True:
            prefix = 'SVD'            
            er = SVDExp(self.reader, self.splitter)            
            er.experiment_dimension(self.splitter.X_train, prefix)       
            n_components = 7
            dataX = er.experiment_best(self.splitter.X_train, n_components) 
            dataX_combined = dataX
            if self.run_final == False:  
                self.experiment_cluster_size(dataX_combined, prefix)
            else:
                n_components = 11
                self.experiment_cluster(dataX_combined, self.splitter.y_train, prefix, 5)

            


    def experiment_bank(self):

        val = 4
        test = False
        
        if self.eval_condition(val, 0, test) == True:
            # without Dim reduction
            prefix = 'No Dim Red'
            n_components = 9   
            if self.run_final == False:
                self.experiment_cluster_size(self.splitter.X_train, prefix)
            else:
                self.experiment_cluster(self.splitter.X_train, self.splitter.y_train, prefix, n_components)


        #With PCA
        if self.eval_condition(val, 1, test) == True:
            prefix = 'PCA'
            er = PCAExp(self.reader, self.splitter)
            er.experiment_dimension()
            n_components = 8   
            dataX = er.experiment_best(n_components)   
            dataX_combined = dataX #np.concatenate((self.splitter.X_train, dataX), axis = 1)   
            if self.run_final == False:  
                self.experiment_cluster_size(dataX_combined, prefix)
            else:
                n_components = 11
                self.experiment_cluster(dataX_combined, self.splitter.y_train, prefix, n_components)
            

         #With ICA
        if self.eval_condition(val, 2, test) == True:
            prefix = 'ICA'
            er = ICAExp(self.reader, self.splitter)
            er.experiment_dimension(self.splitter.X_train)
            n_components = 11      
            dataX = er.experiment_best(n_components)   
            dataX_combined = dataX #np.concatenate((self.splitter.X_train, dataX), axis = 1)   
            
            if self.run_final == False:  
                self.experiment_cluster_size(dataX_combined, prefix)
            else:
                n_components = 9
                self.experiment_cluster(dataX_combined, self.splitter.y_train, prefix, n_components)
            

        #With Random Proj
        if self.eval_condition(val, 3, test) == True:
            prefix = 'RP'            
            er = RandomProjExp(self.reader, self.splitter)
            er.experiment_cluster_size(self.splitter.X_train, prefix)       
            n_components = 14
            dataX = er.experiment_best(n_components) 
            dataX_combined = dataX
            if self.run_final == False:  
                self.experiment_cluster_size(dataX_combined, prefix)
            else:
                n_components = 12
                self.experiment_cluster(dataX_combined, self.splitter.y_train, prefix, 5)


        #With SVD
        if self.eval_condition(val, 4, test) == True:
            prefix = 'SVD'            
            er = SVDExp(self.reader, self.splitter)            
            er.experiment_dimension(self.splitter.X_train, prefix)       
            n_components = 9
            dataX = er.experiment_best(self.splitter.X_train, n_components) 
            dataX_combined = dataX
            if self.run_final == False:  
                self.experiment_cluster_size(dataX_combined, prefix)
            else:
                n_components = 11
                self.experiment_cluster(dataX_combined, self.splitter.y_train, prefix, 5)






    def experiment_cluster_size(self, dataX, prefix):
        print('experiment cluster size KMeans: '+ prefix)
        sum_sq_dist = []
        distortions = []
        db_scores = []
        K = range(3,15)
        X = dataX
        for k in K:
            learner = KMeansLearner(n_clusters = k, random_state = self.random_state)
            kf = learner.fit(X)
            sum_sq_dist.append(kf.inertia_)
            distortions.append(sum(np.min(cdist(X, 
                                        kf.cluster_centers_, 
                            'euclidean'), axis=1)) / X.shape[0])

            kl = learner.fit_predict(X)
            silhouette_avg = silhouette_score(X, kl)
            db_score = davies_bouldin_score(X, kf.labels_)
            db_scores.append(db_score)
            print("For n_clusters =", k,
                    "The average silhouette_score is :", silhouette_avg)
            sample_silhouette_values = silhouette_samples(X, kl)
            #TODO Plot


        
        prefix = self.splitter.reader.dataset + '-' + prefix 
        plot_curve(K, sum_sq_dist, 'Finding Optimal K using Elbow - KMeans',  'K Value', 'Sum of Squared Distances', prefix)
        plot_curve(K, distortions, 'Finding Optimal K using Distortions- KMeans',  'K Value', 'Distortions', prefix)
        plot_curve(K, db_scores, 'Finding Optimal K using DB Scores- KMeans',  'K Value', 'DB Score', prefix)

        """visualizer = KElbowVisualizer(learner, k=(1, 15))
        visualizer.fit(self.splitter.X_train)
        visualizer.show(outpath = 'images\\' + prefix + 'elbow.png')"""


    def experiment_cluster(self, dataX, dataY, prefix, n_clusters):
        #n_clusters = 5
        X = dataX
        learner = KMeansLearner(n_clusters = n_clusters, random_state = self.random_state)
        kmean = learner.fit(X)
        prefix = self.splitter.reader.dataset + '-' + prefix 
        #plot_cluster(X[:], kmean.cluster_centers_, kmean.labels_, n_clusters, 'Clusters Visualization - KMeans', prefix)
        plot_cluster_wo_centroid(X[:],  kmean.labels_, n_clusters, 'Clusters Visualization - KMeans', prefix)
        kmean_labels = learner.fit_predict(X)
        print('cluster output')
        print(kmean_labels)
        print(kmean_labels.shape)
        print('fit labels')
        print(kmean.labels_)
        print(kmean.labels_.shape)        
        hcv = homogeneity_completeness_v_measure(dataY, kmean_labels) 
        print('homogenity/comp/vmeas score of Kmeans: ' + prefix + str(hcv))   
        ari = adjusted_rand_score(dataY, kmean_labels)
        print('ARI score of Kmeans: ' + prefix + str(ari)) 
        return  kmean_labels

    

    def eval_condition(self, val, test_val, test):
        if val == test_val and test == True:
            return True
        elif test == False:
            return True

        return False