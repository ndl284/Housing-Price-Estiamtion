# -------------------------------------------------------------------------------------------------------------------------------
# Program to Implement Ridge and Linear Regression
# author: Noel D. Lobo
# date: 12/15/2016
# -------------------------------------------------------------------------------------------------------------------------------

import pandas
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
from sklearn import preprocessing

# ---------------------------------------------------------------------------------------------------------------------------------
# Method to check if passed value is Nan
# Input
#   ** value - the value that needs to be checked
# Returns
#   ** value or 0 - returns 0 if the value is NaN else returns the original value itself.
# ---------------------------------------------------------------------------------------------------------------------------------
def checkNaN(value):
    if (pandas.isnull(value)):
        return 0
    return value

class Provider:
    # ---------------------------------------------------------------------------------------------------------------------------------
    # Method to perform Linear Regression.
    # Inputs
    #   ** features - Matrix containing all the features extracted from the csv file.
    #   ** labels - Array containing all the labels from the csv file
    #   ** alpha - The learning rate for the ridge regression.
    # Returns
    #   ** w - weight matrix containing the optimized weights after training is complete.
    # ---------------------------------------------------------------------------------------------------------------------------------
    def LinearRegression(self, features, labels, alpha):
        features = preprocessing.normalize(features)                                                                    # normalizing the features
        labels = preprocessing.normalize(labels)                                                                        # normalizing the labels

        labels = labels.T                                                                                               # transposing the labels matrix to a nx1 for matrix multiplication

        l = len(features)
        n = len(features[0])
        features = np.append(features, np.ones(shape=(l,1)),axis=1)                                                     # adding bias to the features

        w = np.random.rand(1, n + 1).T                                                                                  # randomly initializing the weight matrix with value between 0-1
        n = len(labels)

        for i in range(0, 10):                                                                                          # gradient descent
            inner = features.dot(w)                                                                                     # calculating result
            J = np.sum((inner - labels) ** 2)/(2*n)                                                                     # calculating error

            print("Iteration %d, J(w): %f\n" % (i, J))

            df = features.T

            gradient = df.dot(features.dot(w)-labels)/n                                                                 # calculating correnction
            w -= alpha * gradient                                                                                       # updating the weight matrix

        return w

    # ---------------------------------------------------------------------------------------------------------------------------------
    # Method to perform Ridge Regression.
    # Inputs
    #   ** features - Matrix containing all the features extracted from the csv file.
    #   ** labels - Array containing all the labels from the csv file
    #   ** alpha - The learning rate for the ridge regression.
    #   ** penalty - The penalty to be associated with weights features in ridge regression.
    # Returns
    #   ** w - weight matrix containing the optimized weights after training is complete.
    # ---------------------------------------------------------------------------------------------------------------------------------
    def RidgeRegression(self, features, labels, alpha, penalty):
        features = preprocessing.normalize(features)                                                                    # normalizing the features
        labels = preprocessing.normalize(labels)                                                                        # normalizing the labels

        labels = labels.T                                                                                               # transposing the labels to resemble nx1 matrix for matrix multiplication

        l = len(features)
        n = len(features[0])
        features = np.append(features, np.ones(shape=(l, 1)), axis=1)                                                   # for better results adding a bias to the  features

        w = np.random.rand(1, n + 1).T                                                                                  # initializing the weight matrix with random values between 0-1

        n = len(labels)

        for i in range(0, 10):                                                                                          # performing gradient descent for 10 iterations
            inner = features.dot(w)                                                                                     # computing result

            J = np.sum((inner - labels) ** 2) / (2 * n)                                                                 # calculating error

            print("Iteration %d, J(w): %f\n" % (i, J))

            df = features.T

            gradient = df.dot(features.dot(w) - labels) / n                                                             # calculating correction
            lasso = w ** 2
            w = w - (alpha * gradient)-(penalty * lasso)                                                                # updating weights

        return w

    # ---------------------------------------------------------------------------------------------------------------------------------
    # Method runs the Linear and Ridge Regression
    # ---------------------------------------------------------------------------------------------------------------------------------
    def learn(self):
        train_values, train_labels = self.get_data()

        print "\n Linear Regression Implementation\n"                                                                   # Linear Regression
        self.LinearRegression(train_values, train_labels, .1)                                                           # Calling the Linear Regression Method.
        reg = linear_model.LinearRegression()
        print reg.fit(train_values, train_labels).score(train_values, train_labels)

        print "\n Ridge Regression Implementation\n"                                                                    # Ridge Regression
        self.RidgeRegression(train_values, train_labels, .1, 0.1)                                                       # Calling the Ridge Regression Method.
        reg = linear_model.Ridge(alpha=0.25)
        print reg.fit(train_values, train_labels).score(train_values, train_labels)

    # ---------------------------------------------------------------------------------------------------------------------------------
    # Method to get the processed data from the csv files.
    # Returns
    #  **train_final - feature matrix containing all the features of eact property
    #  **train_values - labels matrix containing all the prices of properties with features at the corresponding
    #       index of train_final.
    # ---------------------------------------------------------------------------------------------------------------------------------
    def get_data(self):
        train_labels = []
        train_values =[]

        train_csv = self.get_data_csv()                                                                                 # get all the data from the csv file

        for index in range(1, train_csv.shape[0]):                                                                      # separate the lables and features from the csv data
            train_values.append({train_csv[key][0]: train_csv[key][index] for key in range(0, train_csv.shape[1] - 1)})
            train_labels.append(train_csv[train_csv.shape[1]-1][index])

        train_vector_maker = DictVectorizer()
        train_final = train_vector_maker.fit_transform(train_values).toarray()

        return train_final, train_labels

    # ---------------------------------------------------------------------------------------------------------------------------------
    # Method reads from the csv files, performs initial processing, and returns the data.
    # Returns
    #  **train_csv - two dimensional matrix containing all the features and lables from the csv file.
    # ---------------------------------------------------------------------------------------------------------------------------------
    def get_data_csv(self):
        # Read from csv
        train_csv = pandas.read_csv("train.csv", sep=',', header=None)                                                  # reading from the csv file
        x = 0
        for p in range(1, train_csv.shape[0]):                                                                          #casting the values of the features. Point to be noted here only the
            train_csv[0][p] = float(train_csv[0][p])  # ID                                                               numerical fields are handled the character or text fields are handled in
            train_csv[1][p] = float(train_csv[1][p])  # MSSubClass                                                       get_data() method using sparse dictionary.
            train_csv[3][p] = float(train_csv[3][p])  # LotFrontage
            train_csv[4][p] = float(train_csv[4][p])  # LotArea
            train_csv[17][p] = float(train_csv[17][p])  # OverallQual
            train_csv[18][p] = float(train_csv[18][p])  # OverallCond
            train_csv[19][p] = float(train_csv[19][p])  # YearBuilt
            train_csv[20][p] = float(train_csv[20][p])  # YearRemodAdd
            train_csv[26][p] = float(train_csv[26][p])  # MasVnrArea
            train_csv[34][p] = float(train_csv[34][p])  # BsmtFinSF1
            train_csv[36][p] = float(train_csv[36][p])  # BsmtFinSF2
            train_csv[37][p] = float(train_csv[37][p])  # BsmtUnfSF
            train_csv[38][p] = float(train_csv[38][p])  # TotalBsmtSF
            train_csv[43][p] = float(train_csv[43][p])  # 1stFlrSF
            train_csv[44][p] = float(train_csv[44][p])  # 2stFlrSF
            train_csv[45][p] = float(train_csv[45][p])  # LowQualFinSF
            train_csv[46][p] = float(train_csv[46][p])  # GrLivArea
            train_csv[47][p] = float(train_csv[47][p])  # BsmtFullBath
            train_csv[48][p] = float(train_csv[48][p])  # BsmtHalfBath
            train_csv[49][p] = float(train_csv[49][p])  # FullBath
            train_csv[50][p] = float(train_csv[50][p])  # HalfBath
            train_csv[51][p] = float(train_csv[51][p])  # BedroomAbvGr
            train_csv[52][p] = float(train_csv[52][p])  # KitchenAbvGr
            train_csv[54][p] = float(train_csv[54][p])  # TotRmsAbvGrd
            train_csv[56][p] = float(train_csv[56][p])  # Fireplaces
            train_csv[59][p] = float(train_csv[59][p])  # GarageYrBlt
            train_csv[61][p] = float(train_csv[61][p])  # GarageCars
            train_csv[62][p] = float(train_csv[62][p])  # GarageArea
            train_csv[66][p] = float(train_csv[66][p])  # WoodDeckSF
            train_csv[67][p] = float(train_csv[67][p])  # OpenPorchSF
            train_csv[68][p] = float(train_csv[68][p])  # EnclosedPorch
            train_csv[69][p] = float(train_csv[69][p])  # 3SsnPorch
            train_csv[70][p] = float(train_csv[70][p])  # ScreenPorch
            train_csv[71][p] = float(train_csv[71][p])  # PoolArea
            train_csv[75][p] = float(train_csv[75][p])  # MiscVal
            train_csv[76][p] = float(train_csv[76][p])  # MoSold
            train_csv[77][p] = float(train_csv[77][p])  # YrSold
            train_csv[80][p] = float(train_csv[80][p])  # SalePrice

        for i in range(0,80):                                                                                           # handling NaN values in the data for features that are missing data.
            for j in range(1, train_csv.shape[0]):                                                                      #   replacing them with 0. Handles the NaN values for text features too.
                train_csv[i][j] = checkNaN(train_csv[i][j])

        return train_csv

Provider().learn()
