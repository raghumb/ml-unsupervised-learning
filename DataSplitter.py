from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate

class DataSplitter:
    def __init__(self, reader):
        self.reader = reader
        self.test_size = 0.2
        self.total_num_records = 0
        self.num_splits = 10
        self.X_test = None
        self.y_test = None

    def read_split_data(self):
        X, y, num = self.reader.read_process_data()
        self.total_num_records = num
        # Split dataset into training set and test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size, shuffle = True, random_state=1) 
       
       
        """kf = KFold(n_splits = self.num_splits, random_state = 1, shuffle=True)
        for train_idx, test_idx in kf.split(X_train):
            X_train_cv  = X_train[train_idx]
            y_train_cv  = y_train[train_idx]

            X_test_cv   = X_train[test_idx]
            y_test_cv   = y_train[test_idx]"""
        
        #self.X_cv = ShuffleSplit(n_splits = self.num_splits, test_size = self.test_size, random_state = 1)
        # Use the below to cross validate and get scores, use scoring param
        #scores = cross_validate(learner, X_train, y_train, cv = X_cv, scoring = scoring)  

        train_num_records = self.X_train.shape[0]
        # Account for validation set
        train_num_records = train_num_records * (1 - self.test_size)
        print('train_num_records '+ str(train_num_records))
        train_sizes = []

        
        train_sizes.append(60)
        train_sizes.append(100)
        high_range = False
        range = 400
        if self.reader.dataset == 'Bank':
            if self.reader.type == 'full':
                high_range = True
        
        if high_range == True:
            train_sizes.append(500)
            size = 2000
            range = 2000
        else:
            size = 400
            range = 400



        while(size < train_num_records):
            train_sizes.append(size)
            size = size + range
        # Learning curve training sizes
        #self.learn_train_sizes = train_sizes        
        self.learn_train_sizes = [100, 500, 1000, 1500, 2000, 2500]
        print(self.learn_train_sizes)







    
