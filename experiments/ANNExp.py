import numpy as np
from learner.ANNLearner import ANNLearner
from sklearn import metrics
from experiments.RandomProjExp import RandomProjExp
from experiments.PCAExp import PCAExp
from experiments.ICAExp import ICAExp
from experiments.SVDExp import SVDExp
from sklearn.metrics import accuracy_score, make_scorer
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from experiments.KMeansExp import KMeansExp
from experiments.EMExp import EMExp
from sklearn.preprocessing import OneHotEncoder


class ANNExp:
    def __init__(self, reader, splitter):
        self.reader = reader
        self.splitter = splitter
        #self.expHelper = ExperimentHelper(self.splitter, self.learner)
        self.random_state = 42
        self.prefix = self.splitter.reader.dataset

    
    def split_data(self, dataX, dataY):
        X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, shuffle = True, random_state=1) 
        return X_train, X_test, y_train, y_test

    
    def experiment(self):
        self.experiment_bank()
        self.experiment_clusters()
    
    def experiment_bank(self):
        
        prefix = 'PCA'
        n_components = 8   
        er = PCAExp(self.reader, self.splitter)
        dataX = er.experiment_best_test(n_components, self.splitter.X_test, self.splitter.y_test) 
        X_train, X_test, y_train, y_test = self.split_data(dataX, self.splitter.y_test)
        self.experiment_nn(prefix, X_train, X_test, y_train, y_test)

        prefix = 'ICA'
        n_components = 11
        er = ICAExp(self.reader, self.splitter)
        dataX = er.experiment_best_test(n_components, self.splitter.X_test, self.splitter.y_test)  
        X_train, X_test, y_train, y_test = self.split_data(dataX, self.splitter.y_test)
        self.experiment_nn(prefix, X_train, X_test, y_train, y_test)

        prefix = 'RP'
        n_components = 14  
        er = RandomProjExp(self.reader, self.splitter)
        dataX = er.experiment_best_test(n_components, self.splitter.X_test, self.splitter.y_test)   
        X_train, X_test, y_train, y_test = self.split_data(dataX, self.splitter.y_test)
        self.experiment_nn(prefix, X_train, X_test, y_train, y_test)             


        prefix = 'SVD'
        n_components = 9  
        er = SVDExp(self.reader, self.splitter)
        dataX = er.experiment_best_test(n_components, self.splitter.X_test, self.splitter.y_test)   
        X_train, X_test, y_train, y_test = self.split_data(dataX, self.splitter.y_test)
        self.experiment_nn(prefix, X_train, X_test, y_train, y_test)  

    def experiment_nn(self, prefix, X_train, X_test, y_train, y_test):
        start = time.time()
        self.learner = ANNLearner(
            activation = 'relu',
            alpha = 0.0001,
            hidden_layer_sizes = (50,50,50,50,),
            learning_rate = 'constant',
            solver = 'adam',
            early_stopping = False
        )
        learner_name = 'ANN'
        self.learner.train(X_train, y_train)
        y_pred = self.learner.query(X_test)
        end = time.time()
        diff = abs(end - start)
        print('---------------' + prefix + '-------------------')
        print('time taken for RHC: ' + str(diff))        
        print("Final Accuracy for " + str(learner_name)+": ", 
                        metrics.accuracy_score(y_test, y_pred))   
        print("Confusion matrix for " + str(learner_name)+": ", 
                        confusion_matrix(y_test, y_pred))        
        print("Recall score for " + str(learner_name)+": ", 
                        recall_score(y_test, y_pred))   
        print("Precision score for " + str(learner_name)+": ", 
                        precision_score(y_test, y_pred)) 
        print("f1 score for " + str(learner_name)+": ", 
                        f1_score(y_test, y_pred)) 

        fpr, tpr, thresholds = roc_curve(y_test, y_pred)


    def experiment_clusters(self):
        self.experiment_cluster_km()
        self.experiment_cluster_em()

    def experiment_cluster_km(self):
        km = KMeansExp(self.reader, self.splitter, False)
        n_components = 9
        prefix = 'KM-NN'
        kmean_labels = km.experiment_cluster(self.splitter.X_test, self.splitter.y_test, prefix, n_components)        
        kmean_labels = kmean_labels.reshape(-1,1)
        enc = OneHotEncoder()
        enc.fit(kmean_labels)
        onehotlabels = enc.transform(kmean_labels).toarray()
        print('onehotlabels')
        print(onehotlabels)
        X_train, X_test, y_train, y_test = self.split_data(onehotlabels, self.splitter.y_test)
        self.experiment_nn(prefix, X_train, X_test, y_train, y_test)

        print('combined data')
        prefix = 'KM-NN-combined'
        dataX = np.concatenate((self.splitter.X_test, onehotlabels), axis = 1)   
        X_train, X_test, y_train, y_test = self.split_data(dataX, self.splitter.y_test)
        self.experiment_nn(prefix, X_train, X_test, y_train, y_test)



    def experiment_cluster_em(self):
        km = EMExp(self.reader, self.splitter, False)
        n_components = 9
        prefix = 'EM-NN'
        kmean_labels = km.experiment_cluster(self.splitter.X_test, self.splitter.y_test, prefix, n_components)        
        kmean_labels = kmean_labels.reshape(-1,1)
        enc = OneHotEncoder()
        enc.fit(kmean_labels)
        onehotlabels = enc.transform(kmean_labels).toarray()
        print('onehotlabels')
        print(onehotlabels)
        X_train, X_test, y_train, y_test = self.split_data(onehotlabels, self.splitter.y_test)
        self.experiment_nn(prefix, X_train, X_test, y_train, y_test)

        print('combined data')
        prefix = 'EM-NN-combined'
        dataX = np.concatenate((self.splitter.X_test, onehotlabels), axis = 1)   
        X_train, X_test, y_train, y_test = self.split_data(dataX, self.splitter.y_test)
        self.experiment_nn(prefix, X_train, X_test, y_train, y_test)

                