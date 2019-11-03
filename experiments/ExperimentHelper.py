import numpy as np
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import learning_curve
from plotter import plot_learning_curve
from plotter import plot_model_complexity_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
import pandas as pd


class ExperimentHelper:
    def __init__(self, splitter = None, classifier = None, prefix = None):
        #self.alg = alg
        self.splitter = splitter
        self.classifier = classifier
        self.prefix = prefix
        pass

    #def run_experiment():
    def create_random_seeds(self):
        return np.array([5, 35, 20])
        #return np.random.randint(40, size = 5)

    def learning_curve_exp(self):
        scoring = 'accuracy' #'neg_mean_squared_error'        
        """scores = cross_validate(self.learner.dt, 
                                self.splitter.X_train, 
                                self.splitter.y_train, 
                                cv = self.splitter.X_cv, 
                                scoring = scoring) """

        print('splitter train size '+ str(self.splitter.X_train.shape[0]))
        train_sizes, train_scores, validation_scores = learning_curve( 
                                                        self.classifier,
                                                        self.splitter.X_train,
                                                        self.splitter.y_train,
                                                        train_sizes = self.splitter.learn_train_sizes,
                                                        cv = self.splitter.num_splits,
                                                        scoring = scoring,
                                                        shuffle = True,
                                                        random_state = 1)

        
        
        learner_name = 'ANN'
        print('Training scores:\n\n ', train_scores)
        print('\nValidation scores:\n\n', validation_scores)
        #print("Accuracy for " + str(learner.__class__)+": ", metrics.accuracy_score(y_test, y_pred))

        train_scores_mean = train_scores.mean(axis = 1)
        validation_scores_mean = validation_scores.mean(axis = 1)
        print('Mean training scores\n\n', pd.Series(train_scores_mean, index = train_sizes))
        print('\n', '-' * 20) # separator
        print('\nMean validation scores\n\n',pd.Series(validation_scores_mean, index = train_sizes))
        
        plot_learning_curve(train_sizes, train_scores_mean, validation_scores_mean, learner_name, self.prefix)  

    def model_complexity_exp(self, param_name, param_range):
        scoring = 'accuracy'
        print("in model complexity")        
        learner_name = 'ANN'
        train_scores, validation_scores = validation_curve(
                                            self.classifier,
                                            self.splitter.X_train,
                                            self.splitter.y_train,
                                            param_name,
                                            param_range,
                                            cv = self.splitter.num_splits,
                                            scoring = scoring
                                            )
        
        print('Training scores:\n\n ', train_scores)
        print('\nValidation scores:\n\n', validation_scores)
        #print("Accuracy for " + str(learner.__class__)+": ", metrics.accuracy_score(y_test, y_pred))

        train_mean = train_scores.mean(axis = 1)
        train_std = np.std(train_scores, axis=1)
        validation_mean = validation_scores.mean(axis = 1)
        validation_std = np.std(validation_scores, axis=1)
        #print('MC Mean training scores\n\n', pd.Series(train_scores_mean, index = train_sizes))
        print('\n', '-' * 20) # separator
        #print('\n MC Mean validation scores\n\n',pd.Series(validation_scores_mean, index = train_sizes))
        
        plot_model_complexity_curve(param_name, param_range, train_mean, train_std, validation_mean, validation_std, learner_name, self.prefix, None)        



