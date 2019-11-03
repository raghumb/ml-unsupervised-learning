from learner.EMLearner import EMLearner
from plotter import plot_curve
from experiments.RandomProjExp import RandomProjExp
from experiments.PCAExp import PCAExp
from experiments.ICAExp import ICAExp
#from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
from scipy.spatial.distance import cdist
from experiments.SVDExp import SVDExp
from sklearn.metrics import homogeneity_score
from clusterplotter import plot_cluster, plot_cluster_wo_centroid
from sklearn.metrics import homogeneity_completeness_v_measure, adjusted_rand_score

class EMExp:
    def __init__(self, reader, splitter, run_final):
        self.reader = reader
        self.splitter = splitter
        #self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.random_state = 42
        self.run_final = run_final
        #self.splitter.read_split_data()


    def experiment(self):
        if self.splitter.reader.dataset == 'wine':
            self.experiment_wine()
        else:            
            print('bank flow')
            self.experiment_bank()        
        

    def experiment_wine(self):

        val = 4
        test = False

        # without Dim reduction
        if self.eval_condition(val, 0, test) == True:
            prefix = 'No Dim Red'
            n_components = 7
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
                n_components = 5
                self.experiment_cluster(dataX_combined, self.splitter.y_train, prefix, n_components)

         #With ICA
        if self.eval_condition(val, 2, test) == True:
            prefix = 'ICA'
            er = ICAExp(self.reader, self.splitter)
            #er.experiment_dimension(self.splitter.X_train)
            n_components = 11      
            dataX = er.experiment_best(n_components)   
            dataX_combined = dataX #np.concatenate((self.splitter.X_train, dataX), axis = 1)   
            
            if self.run_final == False:  
                self.experiment_cluster_size(dataX_combined, prefix)
            else:
                n_components = 13
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
                n_components = 5
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
                n_components = 10
                self.experiment_cluster(dataX_combined, self.splitter.y_train, prefix, 5)
            




    def experiment_bank(self):

        val = 4
        test = False
        
        if self.eval_condition(val, 0, test) == True:
            # without Dim reduction
            prefix = 'No Dim Red'
            n_components = 12  
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
                n_components = 12
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
                n_components = 8
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
                n_components = 8
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






    def experiment_cluster_size(self, dataX, prefix):
        bic_scores = []
        aic_scores = []
        K = range(3,15)
        X = dataX
        for k in K:
            learner = EMLearner(n_components = k, random_state = self.random_state)
            kf = learner.fit(X)
            bic_score = kf.bic(X)
            bic_scores.append(bic_score)
            aic_score = kf.aic(X)
            aic_scores.append(aic_score)


        
        prefix = self.splitter.reader.dataset + '-' + prefix 
        plot_curve(K, bic_scores, 'Finding Optimal n using BIC -EM',  '# of Components', 'BIC Score', prefix)
        plot_curve(K, aic_scores, 'Finding Optimal n using AIC - EM',  '# of Components', 'AIC Score', prefix)

        """visualizer = KElbowVisualizer(learner, k=(1, 15))
        visualizer.fit(self.splitter.X_train)
        visualizer.show(outpath = 'images\\' + prefix + 'elbow.png')"""



    def experiment_cluster(self, dataX, dataY, prefix, n_clusters):
        n_components = n_clusters
        X = dataX
        learner = EMLearner(n_components = n_components, random_state = self.random_state)
        kmean = learner.fit(X)
        kmean_labels = learner.fit_predict(X)
        plot_cluster_wo_centroid(X[:], kmean_labels, n_clusters,  'Clusters Visualization - EM', prefix)
        hcv = homogeneity_completeness_v_measure(dataY, kmean_labels) 
        print('homogenity/comp/vmeas score of EM: ' + prefix + str(hcv))   
        ari = adjusted_rand_score(dataY, kmean_labels)
        print('ARI score of EM: ' + prefix + str(ari))    
        return kmean_labels


    def eval_condition(self, val, test_val, test):
        if val == test_val and test == True:
            return True
        elif test == False:
            return True

        return False