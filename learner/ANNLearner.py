from sklearn.neural_network import MLPClassifier
import time

#This code is derived from https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
class ANNLearner:

    def __init__(self, 
                verbose = False, 
                hidden_layer_sizes = (100,),
                activation = 'relu',
                solver = 'adam',
                alpha = 0.0001,
                batch_size = 'auto',
                learning_rate = 'constant',
                learning_rate_init = 0.001,
                power_t = 0.5,
                max_iter = 200,
                shuffle = True,
                random_state = 1,
                tol = 1e-4,                
                warm_start = False,
                momentum = 0.9,
                nesterovs_momentum = True,
                early_stopping = False,
                validation_fraction = 0.1,
                beta_1 = 0.9,
                beta_2 = 0.999,
                epsilon = 1e-8,
                n_iter_no_change = 10):
        self.verbose = verbose
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change  

        self.classifier = MLPClassifier(
            hidden_layer_sizes = self.hidden_layer_sizes,
            activation = self.activation,
            solver = self.solver,
            alpha = self.alpha,
            batch_size = self.batch_size,
            learning_rate = self.learning_rate,
            learning_rate_init = self.learning_rate_init,
            power_t = self.power_t,
            max_iter = self.max_iter,
            shuffle = self.shuffle,
            random_state = self.random_state,
            tol = self.tol,
            verbose = self.verbose,
            warm_start = self.warm_start,
            momentum = self.momentum,
            nesterovs_momentum = self.nesterovs_momentum,
            early_stopping = self.early_stopping,
            validation_fraction = self.validation_fraction,
            beta_1 = self.beta_1,
            beta_2 = self.beta_2,
            epsilon = self.epsilon,
            n_iter_no_change = self.n_iter_no_change 
        )

        pass


    def train(self, X_train, y_train):

        #Create a ann network model
        #hidden_layer_sizes=(30,30,30)
        
        start = time.time()
        #Train the model using the training sets
        self.classifier.fit(X_train, y_train)
        end = time.time()
        diff = abs(end - start)
        print('train time taken for ANN: ' + str(diff))

    def query(self, X_test):
        #Predict the response for test dataset
        start = time.time()
        y_pred = self.classifier.predict(X_test)
        end = time.time()
        diff = abs(end - start)
        print('test time taken for ANN: ' + str(diff))        

        return y_pred