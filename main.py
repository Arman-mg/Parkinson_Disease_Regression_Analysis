# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 23:57:25 2021

@author: Amran MohammadiGilani
"""
"""
Notice: In the second section of the report (Data analysis), it is mentioned that 75% 
of the points is for training data and 25% of the points is test data. However, 
according to the figure captions for SG with Adam method and Gradient Algorithm 
with Minibatches, validation data has also been requested from the student. 
As a result, 70% of the data was allocated to training data, 15% to validation data
and 15% to test data. Also, in the second part of the report and the final table, 
the values of validation data were added.
"""
#%% Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#%% Initialize Classes
# Define The Function to Print & to Plot Vector w_hat
class SolvMinProbl:
    def __init__(self, y, X, y_valid, X_valid, y_test, X_test, mean, stdt):

        # Whole Data Without 'subject#' & 'test_time'
        self.matr = X
        # Np = Number of Patients/Rows & Nf = Number of Features/Columns
        self.Np = X.shape[0]
        self.Nf = X.shape[1]
        # Initializing Test & Validation Data
        self.matr_valid = X_valid
        self.matr_test = X_test
        # Mean & Standard Deviation of Train Data
        self.mean = mean 
        self.stdt = stdt 
        # 'total_UPDRS' as Regressand
        self.vect = y
        self.vect_valid = y_valid
        self.vect_test = y_test
        # Optimum Weights w_hat (Column of Nf Entries)
        self.sol = np.zeros((self.Nf, 1), dtype = float)   
        self.err = 0
        self.mse = 0
        self.stat = 0
        
    #%% Define Mean Square Error
    def MSE_est(self):
        
        self.mse = np.zeros((3, 1), dtype = float)
        # Defination of Train, Validation & Test Data for y with Estimated Ones
        y_train = self.stdt * self.vect + self.mean
        y_train_estimated = self.stdt * np.dot(self.matr, self.sol) + self.mean
        
        y_valid = self.stdt * self.vect_valid + self.mean
        y_valid_estimated = self.stdt * np.dot(self.matr_valid,self.sol) + self.mean
        
        y_test = self.stdt * self.vect_test + self.mean
        y_test_estimated = self.stdt * np.dot(self.matr_test,self.sol) + self.mean
        
        # Calculate MSE for Train, Validation & Test Data
        self.mse[0] = (np.linalg.norm(y_train - y_train_estimated)**2) / self.matr.shape[0]
        self.mse[1] = (np.linalg.norm(y_valid - y_valid_estimated)**2)/self.matr_valid.shape[0]
        self.mse[2] = (np.linalg.norm(y_test - y_test_estimated)**2) / self.matr_test.shape[0]
        
        return self.mse[0], self.mse[1], self.mse[2]
        
    #%% Statistical Properties
    def Stat_prop(self):
        self.stat = np.zeros((3, 2), dtype = float)
        # Defination of Train Data for y with Estimated One
        y_train = self.stdt * self.vect + self.mean
        y_train_estimated = self.stdt * np.dot(self.matr, self.sol) + self.mean
        # Calculate Mean & Standard Deviation of Train Data
        train_err = y_train - y_train_estimated
        self.stat[0][0] = train_err.mean()
        self.stat[0][1] = train_err.std()
        
        # Defination of Validation Data for y with Estimated One
        y_valid = self.stdt * self.vect_valid + self.mean
        y_valid_estimated = self.stdt * np.dot(self.matr_valid, self.sol) + self.mean
        # Calculate Mean & Standard Deviation of Validation Data
        valid_err = y_valid - y_valid_estimated
        self.stat[1][0] = valid_err.mean()
        self.stat[1][1] = valid_err.std()
        
        # Defination of Test Data for y with Estimated One
        y_test = self.stdt * self.vect_test + self.mean
        y_test_estimated = self.stdt * np.dot(self.matr_test, self.sol) + self.mean
        # Calculate Mean & Standard Deviation of Test Data
        test_err = y_test- y_test_estimated
        self.stat[2][0] = test_err.mean()
        self.stat[2][1] = test_err.std()
        
        return self.stat

    #%% Coefficient Determination
    def Coeff_determ(self):
        self.R2 = np.zeros((3, 1), dtype = float)
        # Defination of Train Data for y with Estimated One
        y_train = self.stdt * self.vect + self.mean
        y_train_estimated = self.stdt * np.dot(self.matr, self.sol) + self.mean
        # Evaluate The Performance of a Linear Regression Model
        r2_train = r2_score(y_train , y_train_estimated)
        self.R2[0] = r2_train
        
        # Defination of Validation Data for y with Estimated One
        y_valid = self.stdt * self.vect_valid + self.mean
        y_valid_estimated = self.stdt * np.dot(self.matr_valid, self.sol) + self.mean
        # Evaluate The Performance of a Linear Regression Model
        r2_valid = r2_score(y_valid , y_valid_estimated)
        self.R2[1] = r2_valid
        
        # Defination of Test Data for y with Estimated One
        y_test = self.stdt * self.vect_test + self.mean
        y_test_estimated = self.stdt * np.dot(self.matr_test, self.sol) + self.mean
        # Evaluate The Performance of a Linear Regression Model
        r2_test = r2_score(y_test , y_test_estimated)
        self.R2[2] = r2_test
        
        return self.R2
    
    #%% Plotting Functions
    def plot_w(self, title):
        plt.figure()
        w = self.sol
        n = np.arange(self.Nf)
        plt.plot(n, w)
        # Features Without 'subject#', 'test_time', & 'total_UPDRS'
        features = ['age','sex','motor_UPDRS','Jitter(%)','Jitter(Abs)','Jitter:RAP','Jitter:PPQ5','Jitter:DDP','Shimmer','Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','Shimmer:APQ11','Shimmer:DDA','NHR','HNR','RPDE','DFA','PPE']
        plt.xticks(np.arange(len(features)), features, rotation = 90)
        plt.ylabel('w_hat(n)')
        plt.title(title)
        plt.grid()
        plt.tight_layout()
        plt.savefig(title +'.png', dpi = 800)
        plt.show()
        
    def comparison(self, title, labelx, labely, y, X, mean , stdt):
        plt.figure()
        w = self.sol
        y_estimated = (np.dot(X, w) * stdt) + mean
        # y Has to Be Unnormalized
        y = (y * stdt) + mean
        plt.plot(np.linspace(0, 60), np.linspace(0, 60), 'tab:red')
        plt.scatter(y, y_estimated, s = 5, color = 'tab:blue')
        plt.xlabel(labelx)
        plt.ylabel(labely)
        plt.title(title)
        plt.grid()
        plt.tight_layout()
        plt.savefig(title +'.png', dpi = 800)
        plt.show()
    
    def histogram(self, y, X, y_valid, X_valid, y_test, X_test, mean , stdt, title):
        plt.figure()
        w = self.sol
        # y_train Has to Be Unnormalized
        y_train = (y * stdt) + mean 
        y_train_estimated = (np.dot(X, w) * stdt) + mean
        plt.hist(y_train - y_train_estimated, bins = 100, histtype='bar', edgecolor = 'gray', alpha = 0.7, color = 'cyan', label = "Training")
        
        # y_valid Has to Be Unnormalized
        y_valid = (y_valid * stdt) + mean
        y_valid_estimated = (np.dot(X_valid, w) * stdt) + mean    
        plt.hist(y_valid - y_valid_estimated, bins = 100, histtype='bar', edgecolor = 'gray', alpha = 0.7, color = 'orange', label="Validation")
        
        # y_test Has to Be Unnormalized
        y_test = (y_test * stdt) + mean 
        y_test_estimated = (np.dot(X_test, w) * stdt) + mean        
        plt.hist(y_test - y_test_estimated, bins = 100, histtype='bar', edgecolor = 'gray', alpha = 0.7, color = 'crimson', label = "Test")     
              
        plt.xlabel('Estimation Error')
        plt.ylabel('Frequency')
        plt.title('Histogram for the Estimation Error for Training, Validation & Test Data' + title)
        plt.legend(loc = 'best')
        plt.tight_layout()
        plt.savefig(title +'.png', dpi = 800)
        plt.show()
      
    def plot_err(self, title, logy = 1, logx = 0):
        err = self.err
        plt.figure()
        # Train Data
        if(logy == 0) & (logx == 0):
            plt.plot(err[:, 0], err[:, 1], color = 'orchid', label = 'Train')
        if (logy == 1) & (logx == 0):
            plt.semilogy(err[:, 0], err[:, 1], color = 'orchid', label = 'Train')
        if (logy == 0) & (logx == 1):
            plt.semilogx(err[:, 0], err[:, 1], color = 'orchid', label = 'Train')
        if (logy == 1) & (logx == 1):
            plt.loglog(err[:, 0], err[:, 1], color = 'orchid', label = 'Train')
            # Validation Data
        if (logy == 0) & (logx == 0):
            plt.plot(err[:, 0], err[:, 2], alpha = 0.7, color = 'teal', label = 'Validation')
        if (logy == 1) & (logx == 0):
            plt.semilogy(err[:, 0], err[:, 2], alpha = 0.7, color = 'teal', label = 'Validation')
        if (logy == 0) & (logx == 1):
            plt.semilogx(err[:, 0], err[:, 2], alpha = 0.7, color = 'teal', label = 'Validation')
        if (logy == 1) & (logx == 1):
            plt.loglog(err[:, 0], err[:, 2], alpha = 0.7, color = 'teal', label = 'Validation')
        plt.xlabel('n')
        plt.ylabel('e(n)')
        plt.legend()
        plt.title(title)
        # Leave Some Space Between The Max/Min Value and The Frame of The Plot
        plt.margins(0.01, 0.1)
        plt.grid()
        plt.tight_layout()
        plt.savefig(title + '.png', dpi = 800)
        plt.show()

#%% LLS method
class LLS(SolvMinProbl):
    def run(self):
        
        X = self.matr
        X_valid = self.matr_valid
        X_test = self.matr_test
        
        y = self.vect
        y_valid = self.vect_valid
        y_test = self.vect_test
      
        w = np.random.rand(self.Nf, 1)
        w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
        self.sol = w
        
        self.err = np.zeros((1, 4), dtype = float)
        
        # Errors on Standardized Vectors
        self.err[0, 1] = np.linalg.norm(y - np.dot(X, w))**2 / X.shape[0]
        self.err[0, 2] = np.linalg.norm(y_valid - np.dot(X_valid, w)) ** 2 / X_valid.shape[0]
        self.err[0, 3] = np.linalg.norm(y_test - np.dot(X_test, w)) ** 2 / X_test.shape[0]
        return self.err[0, 1], self.err[0, 2], self.err[0, 3]
          
#%% Stochastic Gradient with Adam method
class SGwA(SolvMinProbl):
    def run(self, y_valid, X_valid, y_test, X_test, learning_rate, beta1, beta2, iterations, epsilon):
        
        X = self.matr
        X_valid = self.matr_valid
        X_test = self.matr_test
        
        y = self.vect
        y_valid = self.vect_valid
        y_test = self.vect_test
        
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.iterations = iterations
        self.epsilon = epsilon 
        
        w = np.random.rand(self.Nf, 1) # Random Samples From a Uniform Distribution
        self.MSE = np.zeros((iterations,4), dtype = float)
        self.err = np.zeros((iterations,4), dtype = float)
        
        m_t = 0 # 1st Moment Vector
        m_t_hat = 0 
        v_t = 0 # 2nd Moment Vector
        v_t_hat = 0
        stop = 0
        
        for t in range(iterations):
            if(stop < 50):
                grad = 2*X.T@(X@w-y)
                m_t = beta1 * m_t + (1 - beta1) * grad
                v_t = beta2 * v_t + (1 - beta2) * np.power(grad, 2)
                # Calculate The Bias-corrected Estimates
                m_t_hat = m_t / (1 - np.power(beta1, t + 1))
                v_t_hat = v_t / (1 - np.power(beta2, t + 1))
                # Updates The Parameter
                w = w -learning_rate * m_t_hat / (np.sqrt(v_t_hat) + epsilon)
                
                self.MSE[t,0] = t
                self.MSE[t,1] = np.linalg.norm(y - np.dot(X,w))**2 / X.shape[0]
                self.MSE[t,2] = np.linalg.norm(y_valid - X_valid@w)**2 / X_valid.shape[0]
                self.MSE[t,3] = np.linalg.norm(y_test - X_test@w)**2 / X_test.shape[0]
                
                if (self.MSE[t,2]>self.MSE[t-1,2]):
                    stop += 1
                else:
                    stop = 0
            else:
                print('The number of iterations is :',t)
                break
            
        self.sol = w
        self.err = self.MSE[0:t]
                           
        return self.err[-1,1], self.err[-1,2], self.err[-1,3]

#%% Gradient Algorithm with Minibatches
class GAM(SolvMinProbl):
    def run(self, y_valid, X_valid, y_test, X_test, learning_rate, iterations, minibatch_size):
       
        X = self.matr
        X_valid = self.matr_valid
        X_test = self.matr_test
        
        y = self.vect
        y_valid = self.vect_valid
        y_test = self.vect_test
        
        self.learning_rate = learning_rate
        self.iterations = iterations
        
        self.minibatch_size = minibatch_size
        self.batch_num = self.Np // self.minibatch_size
        
        w = np.random.rand(self.Nf, 1)
        self.MSE = np.zeros((iterations,4), dtype = float)
        self.err = np.zeros((iterations,4), dtype = float)
        
        stop = 0
        
        for t in range(iterations):
            if (stop < 50):
                for b in range(self.batch_num):
                    if Np % minibatch_size != 0 & b == self.batch_num - 1:
                        Xi = X_train[b*minibatch_size:]
                        yi = y_train[b*minibatch_size]   
                    else:
                        Xi = X_train[b*minibatch_size:(b + 1) * minibatch_size]
                        yi = y_train[b*minibatch_size:(b + 1) * minibatch_size]
  
                    grad = 2 * np.dot(Xi.T, (np.dot(Xi, w) - yi))
                    w = w - learning_rate * grad
                    
                    self.MSE[t,0] = t
                    self.MSE[t,1] = np.linalg.norm(y - np.dot(X, w))**2 / X.shape[0]
                    self.MSE[t,2] = np.linalg.norm(y_valid - np.dot(X_valid, w)) ** 2 / X_valid.shape[0]
                    self.MSE[t,3] = np.linalg.norm(y_test - np.dot(X_test, w)) ** 2 / X_test.shape[0]
                    
                if (self.MSE[t,2]>=self.MSE[t-1,2]):
                    stop += 1
                else:
                    stop = 0
                    
            else:
                print('The number of iterations is :',t)
                break
         
        self.sol = w
        self.err = self.MSE[0:t]    
            
        return self.err[-1,1], self.err[-1,2], self.err[-1,3]
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     MAIN CODE     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__ == '__main__':

# Prepare and Analyze the Data
    plt.close('all')

    # Checking Data
    x = pd.read_csv("data/parkinsons_updrs.csv")
    x.info()

    Features = list(x.columns)
    print("The List of Features is: ", Features)

    # Introducing New Data by Deleting Unwanted Items
    X = x.drop(['subject#' , 'test_time'], axis = 1)
    # Np = number of patients/rows & Nf = number of features/columns
    Np , Nf = X.shape
    Features = list(X.columns)
    print("The New Features List is: ", Features)

    #%% Correlation
    #Normalizing Data
    Xnorm = (X-X.mean())/X.std()

    # Measure The Covariance
    Covariance = Xnorm.cov()

    # Plot Covariance Matrix of The Features
    plt.figure()
    plt.matshow(np.abs(Covariance.values), fignum = 0, cmap = 'YlGnBu')
    plt.xticks(np.arange(len(Features)), Features, rotation = 90)
    plt.yticks(np.arange(len(Features)), Features, rotation = 0)
    plt.colorbar()
    plt.title("Covariance Matrix of The Features")
    plt.tight_layout()
    plt.savefig('./Correlation Coefficient Between Features.png', dpi = 800)
    plt.show()

    # Plot Correlation Coefficient Between motor_UPDRS & other Features
    plt.figure()
    Covariance.motor_UPDRS.plot()
    plt.grid()
    plt.xticks(np.arange(len(Features)), Features, rotation = 90)
    plt.title("Correlation Coefficient Between motor_UPDRS & other Features")
    plt.tight_layout()
    plt.savefig('./Correlation Coefficient Between motor_UPDRS & other Features.png', dpi = 800)
    plt.show()

    # Plot Correlation Coefficient Between total_UPDRS & other Features
    plt.figure()
    Covariance.total_UPDRS.plot()
    plt.grid()
    plt.xticks(np.arange(len(Features)), Features, rotation = 90)
    plt.title("Correlation Coefficient Between total_UPDRS & other Features")
    plt.savefig('./Correlation Coefficient Between total_UPDRS & other Features.png', dpi = 800)
    plt.show()

    #%% Create Shuffled Data
    np.random.seed(301000) # My IDnumber is s301000
    indexsh = np.arange(Np)
    np.random.shuffle(indexsh)
    X_shuffled = X.copy(deep=True)
    X_shuffled = X_shuffled.set_axis(indexsh, axis = 0, inplace = False)
    X_shuffled = X_shuffled.sort_index(axis = 0)

    #%% Create Training, Validation & Test Data
    N_train = int(Np*0.70)
    N_valid = int((Np - N_train)*0.50)
    N_test = Np - N_train - N_valid
    # Desired Feature as 'total_UPDRS'
    des_F = 3
    #%% Data Standard Deviation & Mean
    data_train = X_shuffled[0:N_train]
    data_valid = X_shuffled[N_train:N_train + N_valid]
    data_test = X_shuffled[N_train + N_valid:]
     
   
    # Feature's Mean & Standard Deviation 
    Mean = np.mean(data_train.values, 0) 
    Standard_Dev = np.std(data_train.values, 0)
    
    data_train_norm = (data_train.values - Mean) / Standard_Dev
    data_valid_norm = (data_valid.values - Mean) / Standard_Dev
    data_test_norm = (data_test.values - Mean) / Standard_Dev

    #%% Data Subsets
    y_train = data_train_norm[:, des_F]  
    y_train = np.reshape(y_train,(4112,1))
    X_train = np.delete(data_train_norm, des_F, 1)

    y_test = data_test_norm[:, des_F]
    y_test = np.reshape(y_test,(882,1))
    X_test = np.delete(data_test_norm, des_F, 1) 

    y_valid = data_valid_norm[:, des_F]
    y_valid = np.reshape(y_valid,(881,1))
    X_valid = np.delete(data_valid_norm, des_F, 1)

    
    #%% Method Preparation
    # Statistical Properties for Error (Rows: Train, Validation, Test / Columns: Mean & Standard Deviation)
    LLS_stat = np.zeros((3, 2), dtype = float)
    SGwA_stat = np.zeros((3, 2), dtype = float)
    GAM_stat = np.zeros((3, 2), dtype = float)
    
    MSE_train = np.zeros((3, 1), dtype = float)   
    MSE_valid = np.zeros((3, 1), dtype = float)
    MSE_test = np.zeros((3, 1), dtype = float)
    
    # MSE Matrix for Values (Rows: MSE_train, MSE_valid, MSE_test / Columns: LLS, SGwA, GAM)
    MSE_matr = np.zeros((3, 3), dtype = float) 
    
    # R^2 (Coefficient of Determination) Regression Score Function
    R2 = np.zeros((3,3), dtype = float)
    
    # Plotting Preparation
    logy = 1
    logx = 0
    
    #%% LLS Method
    a = LLS(y_train, X_train, y_valid, X_valid, y_test, X_test, Mean[des_F], Standard_Dev[des_F])
    MSE_train[0], MSE_valid[0], MSE_test[0] = a.run()
    MSE_matr[:,0] = a.MSE_est()
    LLS_stat = a.Stat_prop()
    a.plot_w('Optimum Weight Vector - LLS')
    # Extracting Mean & Variance of The 'total_UPDRS'
    a.comparison('y_train_estimated vs y_train - LLS','y_train_estimated','y_train', y_train, X_train, Mean[des_F], Standard_Dev[des_F]) 
    a.comparison('y_test_estimated vs y_test - LLS','y_test_estimated','y_test', y_train, X_train, Mean[des_F], Standard_Dev[des_F])  
    a.histogram(y_train, X_train, y_valid, X_valid, y_test, X_test, Mean[des_F], Standard_Dev[des_F], ' - LLS')
    R2 = a.Coeff_determ()
    
    #%% SGwA Method
    iterations = 25000 # The Number of SG Round Before Stopping The Optimization
    learning_rate = 0.001 # The Amount of Learning Happening at Each Time Step
    beta1 = 0.9 # First Decaying Average With Proposed Default Value of 0.9 
    beta2 = 0.999 # Second Decaying Average With Proposed Default Value of 0.999
    epsilon = 1e-8 # A Variable for Numerical Stability During The Division
    
    b = SGwA(y_train, X_train, y_valid, X_valid, y_test, X_test, Mean[des_F], Standard_Dev[des_F])
    MSE_train[1], MSE_valid[1], MSE_test[1] = b.run(y_valid, X_valid, y_test, X_test, learning_rate, beta1, beta2, iterations, epsilon)
    MSE_matr[:,1] = b.MSE_est()

    b.plot_w('Optimum Weight Vector - SGwA')
    b.plot_err('Square Error: Stochastic Gradient with Adam', logy, logx)
    # Extracting Mean & Variance of The 'total_UPDRS
    b.comparison('y_test_estimated vs y_test - SGwA','y_test_estimated','y_test', y_test, X_test, Mean[des_F], Standard_Dev[des_F])
    b.histogram(y_train, X_train, y_test, X_test, y_valid, X_valid, Mean[des_F], Standard_Dev[des_F], ' - SGwA' )
    SGwA_stat = b.Stat_prop()
    R2 = b.Coeff_determ()
    
    #%% SGwA Method
    minibatch_size = 32
    
    c = GAM(y_train, X_train, y_valid, X_valid, y_test, X_test, Mean[des_F], Standard_Dev[des_F])
    MSE_train[2], MSE_valid[2], MSE_test[2] = c.run(y_valid, X_valid, y_test, X_test, learning_rate, iterations, minibatch_size)
    MSE_matr[:,2] = c.MSE_est()

    c.plot_w('Optimum Weight Vector - GAM')
    # Extracting Mean & Variance of The 'total_UPDRS
    c.comparison('y_test_estimated vs y_test - GAM','y_test_estimated','y_test', y_test, X_test, Mean[des_F], Standard_Dev[des_F])
    c.histogram(y_train, X_train, y_test, X_test, y_valid, X_valid, Mean[des_F], Standard_Dev[des_F], ' - GAM' )
    GAM_stat = c.Stat_prop()
    R2 = c.Coeff_determ()