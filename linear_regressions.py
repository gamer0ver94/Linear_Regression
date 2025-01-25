import pandas as pd
from car import Car
import numpy as np
import matplotlib.pyplot as plt
class LinearRegressions:
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
        for data in self.data:
            data.km = (data.km - independent_mean) / independent_deviation
            data.price = (data.price - dependent_mean) / dependent_deviation
        print('deviation: ' + str(dependent_deviation) + '\nmean: ' + str(dependent_mean))
    # calculate the variance of a list of numbers
    def calculate_variance(self, data):
        m = len(data)
        error = 0
        mean = np.mean(data)
        for value in data:
            error += (value - mean) ** 2
        return np.sqrt(error / (m - 1))
    
    # predict price base on theta1 and theta2
    def predict_price(self, km, weight, bias):
        return bias + (weight * km)

    # perform gradient descedent algo
    def gradient_descedent(self, weight,bias, learning_rate, error_history):
        
        deriv_weight = self.calc_derivative_weight(weight,bias)
        deriv_bias = self.calc_derivative_bias(weight, bias)
        weight = weight - (learning_rate * deriv_weight)
        bias = bias - (learning_rate * deriv_bias)
        
        return weight, bias
    
    def calc_derivative_weight(self, weight, bias):
        error = 0
        for data in self.data:
            predicted_price = self.predict_price(data.km, weight, bias)
            error += (predicted_price - data.price) * (2 * data.km)
        return error / len(self.data)


    def calc_derivative_bias(self, weight, bias):
        error = 0
        for data in self.data:
            predicted_price = self.predict_price(data.km, weight, bias)
            error += (predicted_price - data.price) * 2
        return error / len(self.data)

    def mse(self, weight, bias):
        mse = 0
        for data in self.data:
            predicted_price = self.predict_price(data.km, weight, bias)
            mse += (predicted_price - data.price) ** 2
        return mse / len(self.data)

    # train model using learning rate to define calculation rate
    def train_model(self, error_history):
        weight, bias = 0,0
        self.extract_data()
        for i in range(1000):
            error_history.append(self.mse(weight, bias))
            weight, bias = self.gradient_descedent(weight, bias, self.learning_rate, error_history)
            # print('theta : ' + str(theta))
        print(error_history)
        #self.showGraphData(theta, error_history)
        plt.plot(error_history)
        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error')
        plt.title('Error History During Training')
        plt.show()
        return weight, bias
    
    def showGraphData(self, theta, error_history):

        # Extract mileage (x) and price (y) from the data
        # x = [float(car.km) for car in self.data]
        # y = [float(car.price) for car in self.data]
        x = []
        for i in range(len(error_history)):
            x.append(i)

        # Create the scatter plot for the data points
        plt.plot(x, error_history, color='red', label="Data points")


        # Adding labels and legend
        plt.xlabel('Iter (km)')
        plt.ylabel('error')
        plt.legend()

        # Show the plot
        plt.show()
