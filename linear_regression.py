import pandas as pd
from car import Car
import numpy as np
class LinearRegression:
    # initialize constructor
    def __init__(self, file_path, learning_rate):
        self.file_path = file_path
        self.data_size = 0
        self.learning_rate = learning_rate
        self.data = []
    # extract data from file name
    def extract_data (self):
        file_data = pd.read_csv(self.file_path)
        self.data_size = len(file_data)
        for i in range(self.data_size):
            km = file_data.km[i]
            price = file_data.price[i]
            self.data.append(Car(km, price))
        self.standardideze_data()
        #self.print_data()

    def print_data(self):
        for data in self.data:
            print( '  ' + str(data.price))
    # standardization data
    def standardideze_data(self):
        independent_var = []
        independent_mean = 0
        dependent_var = []
        dependent_mean = 0
        for data in self.data:
            dependent_var.append(data.price)
            independent_var.append(data.km)
        dependent_mean = np.mean(dependent_var)
        independent_mean = np.mean(independent_var)
        independent_deviation = self.calculate_variance(independent_var)
        dependent_deviation = self.calculate_variance(dependent_var)
        #print(str(dependent_deviation) + ' ' +str(dependent_mean))
        for data in self.data:
            data.km = (data.km - independent_mean) / independent_deviation
            data.price = (data.price - dependent_mean) / dependent_deviation
    
    # calculate the variance of a list of numbers
    def calculate_variance(self, data):
        m = len(data)
        error = 0
        mean = np.mean(data)
        for value in data:
            error += (value - mean) ** 2
        return np.sqrt(error / (m - 1))
    
    # predict price base on theta1 and theta2
    def predict_price(self, km, theta):
        return theta[0] + (theta[1] * km)

    # perform gradient descedent algo

    def gradient_descedent(self, theta, learning_rate):
        new_theta = [0,0]
        new_theta[0] = (theta[0] - learning_rate * (self.mse(theta, 0)) / len(self.data))
        new_theta[1] = (theta[1] - learning_rate * (self.mse(theta, 1)) / len(self.data))
        return new_theta

    def mse(self, theta, test):
        error = 0
        if (test == 0):
            for data in self.data:
                error += (self.predict_price(data.km, theta) - data.price)
        else:
            for data in self.data:
                error += (self.predict_price(data.km, theta) - data.price) * data.km
        return error



    # train model using learning rate to define calculation rate
    def train_model(self):
        theta = [0,0]
        self.extract_data()
        for i in range(1000):
            theta = self.gradient_descedent(theta, self.learning_rate)
        return theta