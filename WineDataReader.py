import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg') 			  		 			 	 	 		 		 	  		   	  			  	
import matplotlib.pyplot as plt
from plotter import plot_scatter
from sklearn.preprocessing import MinMaxScaler

class WineDataReader:

    def __init__(self):
        self.data_file = "winequality-white.csv"
        self.dataset = 'wine'
        pass

    def read_process_data(self):
        self.read()
        self.clean_data()
        #self.explore_data()
        #self.explore_scatter()
        #self.explore_data_plots()
        X, y = self.pre_process()
        return X, y, self.total_num_records
        


    def read(self):
        col_names = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]
        self.data = pd.read_csv("data/" + self.data_file, header=0, names=col_names)
        self.total_num_records = self.data.shape[0]
        print('Total records ' + str(self.total_num_records))

    def clean_data(self):
        #value = self.data['quality']
        self.data['quality'] = np.where(self.data['quality'] > 6, 1, 0)
        """if self.data['quality'] > 6:
            self.data['quality'] = 1
        else:
            self.data['quality'] = 0"""

        print(self.data)

    
    def explore_data(self):
        """value = self.data['housing']
        #plt.hist([job], color=['orange'], xrot = 90)
        
        plt.hist([value], color=['orange'])
        plt.xlabel("Job")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.75)
        #ax.set_xticklabels(ax.get_xticklabels(),rotation=60)
        plt.xticks(rotation = 90)
        plt.savefig('housing.png')"""

        print('correlation coeff')
        corr  = self.data.corr(method ='pearson') 
        pd.set_option('display.max_columns', 50)
        print(corr)   

    def explore_scatter(self):
        #feature_cols = ["fixed acidity","residual sugar","chlorides","total sulfur dioxide","density","alcohol"]
        feature_cols = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]
        fa = self.data['fixed acidity']
        rs = self.data['residual sugar']
        c = self.data['chlorides']
        tso = self.data['total sulfur dioxide']
        d = self.data['density']
        a = self.data['alcohol']
        """plot_scatter(fa, rs, 'Fixed Acidity', 'Residual Sugar', 'Fixed Acidity vs Residual Sugar')
        plot_scatter(c, tso, 'Chlorides', 'Total Sulfur Dioxide', 'Chlorides vs total sulfur dioxide')
        plot_scatter(d, a, 'Density', 'Alcohol', 'Density vs Alcohol')
        plot_scatter(fa, a, 'Fixed Acidity', 'Alcohol', 'Fixed Acidity vs Alcohol')
        plot_scatter(rs, a, 'Residual Sugar', 'Alcohol', 'Residual Sugar vs Alcohol')"""
    
    def explore_data_plots(self):
        fa = self.data['fixed acidity']
        print(type(fa))
        print(fa.size)
        plt.hist(fa, color=['orange'], bins='auto') 
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.75) 
        plt.title('Fixed Acidity Histogram')       
        #plt.xticks(rotation = 90)
        plt.savefig('images/fixedacidity.png')
        plt.clf()

        fa = self.data['residual sugar']
        plt.hist(fa, color=['orange']) 
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.75)    
        plt.title('Residual Sugar Histogram')     
        #plt.xticks(rotation = 90)
        plt.savefig('images/residualsugar.png')
        plt.clf()  

        fa = self.data['chlorides']
        plt.hist(fa, color=['orange']) 
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title('Chlorides Histogram') 
        plt.grid(axis='y', alpha=0.75)        
        #plt.xticks(rotation = 90)
        plt.savefig('images/chlorides.png')
        plt.clf()   

        fa = self.data['total sulfur dioxide']
        plt.hist(fa, color=['orange']) 
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title('Total Sulfur Dioxide Histogram') 
        plt.grid(axis='y', alpha=0.75)        
        #plt.xticks(rotation = 90)
        plt.savefig('images/totalsulfurdioxide.png')
        plt.clf()   

        fa = self.data['density']
        plt.hist(fa, color=['orange'])  
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title('Density Histogram') 
        plt.grid(axis='y', alpha=0.75)        
        #plt.xticks(rotation = 90)
        plt.savefig('images/density.png')
        plt.clf()  

        fa = self.data['alcohol']
        plt.hist(fa, color=['orange']) 
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title('Alcohol Histogram') 
        plt.grid(axis='y', alpha=0.75)        
        #plt.xticks(rotation = 90)
        plt.savefig('images/alcohol.png')
        plt.clf()                          

        """print('correlation coeff')
        corr  = self.data.corr(method ='pearson') 
        pd.set_option('display.max_columns', 50)
        print(corr) """


    def pre_process(self):
        #feature_cols = ["fixed acidity","residual sugar","chlorides","total sulfur dioxide","density","alcohol"]
        feature_cols = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]
        #feature_cols = ["fixed acidity","residual sugar","total sulfur dioxide","alcohol"]
            
        X = self.data[feature_cols]
        y = self.data.quality
        scaler = MinMaxScaler(feature_range=[0, 1])
        X = scaler.fit_transform(X)
        #scaler = preprocessing.StandardScaler().fit(X)
        #X = scaler.transform(X)

        #print(X)
        return X, y

    
