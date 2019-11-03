import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg') 			  		 			 	 	 		 		 	  		   	  			  	
import matplotlib.pyplot as plt
from plotter import plot_scatter
from sklearn.preprocessing import MinMaxScaler

class BankDataReader:

    def __init__(self):
        self.data_file = "bank-processed-new.csv"
        #self.data_file = "bank-full.csv"
        self.dataset = 'bank'
        self.type = 'full'
        pass

    def read_process_data(self):
        self.read()
        #self.explore_data()
        X, y = self.pre_process()
        #self.corr_coef(X)
        #self.explore_scatter(X)
        #self.explore_data_plots(X)
        return X, y, self.total_num_records
        


    def read(self):
        col_names = ['age', 'job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','y']
        self.data = pd.read_csv("data/" + self.data_file, header=0, names=col_names)
        self.total_num_records = self.data.shape[0]
        print('Total records ' + str(self.total_num_records))

    
    def explore_scatter(self, X):
        feature_cols = ['age','job','marital','education','default','loan','housing','contact','poutcome']
        age = X['age']
        job = X['job']
        marital = X['marital']
        education = X['education']
        housing = X['housing']
        contact = X['contact']
        loan = X['loan']
        plot_scatter(age, job, 'Age', 'Job', 'Age vs Job')
        plot_scatter(marital, education, 'marital', 'education', 'marital vs education')
        plot_scatter(housing, contact, 'housing', 'contact', 'housing vs contact')
        plot_scatter(loan, education, 'loan', 'education', 'loan vs education')
        plot_scatter(job, marital, 'job', 'marital', 'job vs marital')
    
    
    def explore_data_plots(self, X):
    
        fa = X['age']
        print(type(fa))
        print(fa.size)

        plt.hist(fa, color='orange', bins='auto') 
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.75) 
        plt.title('Age Histogram')       
        #plt.xticks(rotation = 90)
        plt.savefig('images/age.png')
        plt.clf()

        fa1 = X['job']
        print(type(fa1))
        print(fa1.size)
        print(fa1)
        plt.hist(fa1, color='orange', bins='auto') 
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.75) 
        plt.title('Job Histogram')       
        #plt.xticks(rotation = 90)
        plt.savefig('images/job.png')
        plt.clf()

        fa = X['marital']
        plt.hist(fa, color='orange', bins='auto') 
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.75) 
        plt.title('Marital Histogram')       
        #plt.xticks(rotation = 90)
        plt.savefig('images/marital.png')
        plt.clf()   

        fa = X['education']
        plt.hist(fa, color='orange', bins='auto') 
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.75) 
        plt.title('Education Histogram')       
        #plt.xticks(rotation = 90)
        plt.savefig('images/education.png')
        plt.clf() 

        fa = X['loan']
        plt.hist(fa, color='orange', bins='auto') 
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.75) 
        plt.title('Loan Histogram')       
        #plt.xticks(rotation = 90)
        plt.savefig('images/loan.png')
        plt.clf() 


        fa = X['housing']
        plt.hist(fa, color='orange', bins='auto') 
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.75) 
        plt.title('Housing Histogram')       
        #plt.xticks(rotation = 90)
        plt.savefig('images/housing.png')
        plt.clf() 

        fa = X['contact']
        plt.hist(fa, color='orange', bins='auto') 
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.75) 
        plt.title('Contact Histogram')       
        #plt.xticks(rotation = 90)
        plt.savefig('images/contact.png')
        plt.clf()  

        fa = X['poutcome']
        plt.hist(fa, color='orange', bins='auto') 
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.75) 
        plt.title('Poutcome Histogram')       
        #plt.xticks(rotation = 90)
        plt.savefig('images/poutcome.png')
        plt.clf()                                               








    def corr_coef(self, X):
        print('correlation coeff')
        corr  = X.corr(method ='pearson') 
        pd.set_option('display.max_columns', 50)
        print(corr)         
    
    def explore_data(self):
        value = self.data['housing']
        #plt.hist([job], color=['orange'], xrot = 90)
        
        plt.hist([value], color=['orange'])
        plt.xlabel("Job")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.75)
        #ax.set_xticklabels(ax.get_xticklabels(),rotation=60)
        plt.xticks(rotation = 90)
        plt.savefig('housing.png')

        print('correlation coeff')
        corr  = self.data.corr(method ='pearson') 
        pd.set_option('display.max_columns', 50)
        print(corr)        

 
    def sample_data(self):
        if self.type == 'full':
            print("Sampling full data")
            #self.data1 = self.data.sample(frac=0.25, replace=True, random_state=1)
            #self.data['y'] = np.where(self.data['y'] == 'yes')
            
            data1 = self.data[self.data['y']=='yes'].sample(n=3103)
            data2 = self.data[self.data['y']=='no'].sample(n=8089)
            data3 = data1.append(data2)
            self.data = data3.sample(frac=1)

            self.data.to_csv('bank-processed.csv', index=False)        
 
    def pre_process(self):
        feature_cols = ['age', 'job','marital','education','default','balance','housing','loan','contact','duration','campaign','previous','poutcome']
        #feature_cols = ['age','job','marital','education','default','loan','housing','contact','poutcome']
        X = self.data[feature_cols]
        le_y = preprocessing.LabelEncoder()
        le_y.fit(self.data['y'])
        self.data['y'] = le_y.transform(self.data['y'])
        y = self.data.y
        X = self.encode_features(X)
        scaler = MinMaxScaler(feature_range=[0, 1])
        X = scaler.fit_transform(X)
        #print(X)
        return X, y

    def encode_features(self, X):
        le = preprocessing.LabelEncoder()
        le.fit(X['job'])
        X['job'] = le.transform(X['job'])
    
        le_m = preprocessing.LabelEncoder()
        le_m.fit(X['marital'])
        X['marital'] = le_m.transform(X['marital'])
        le_l = preprocessing.LabelEncoder()
        le_l.fit(X['loan'])
        X['loan'] = le_l.transform(X['loan'])

        le_d = preprocessing.LabelEncoder()
        le_d.fit(X['default'])
        X['default'] = le_d.transform(X['default'])        
        
        le_c = preprocessing.LabelEncoder()
        le_c.fit(X['contact'])
        X['contact'] = le_c.transform(X['contact'])

        le_e = preprocessing.LabelEncoder()
        le_e.fit(X['education'])
        X['education'] = le_e.transform(X['education'])

        le_h = preprocessing.LabelEncoder()
        le_h.fit(X['housing'])
        X['housing'] = le_h.transform(X['housing'])  

        le_p = preprocessing.LabelEncoder()
        le_p.fit(X['poutcome'])
        X['poutcome'] = le_p.transform(X['poutcome']) 
         
        return X      


