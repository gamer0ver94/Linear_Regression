import pandas as pd
from utils.car import Car
import numpy as np
import matplotlib.pyplot as plt
class LinearRegressions:
    # initialize constructor
    def __init__(self, file_path, learning_rate):
        self.file_path = file_path
        self.data_size = 0
        self.learning_rate = learning_rate
        self.data = []
        self.theta_history = []
    # extract data from file name
    def extract_data (self):
        file_data = pd.read_csv(self.file_path)
        self.data_size = len(file_data)
        for i in range(self.data_size):
            km = file_data.km[i]
            price = file_data.price[i]
            self.data.append(Car(km, price))
        return self.standardize_data()

    def print_data(self):
        for data in self.data:
            print( '  ' + str(data.price))
    # standardization data
    def standardize_data(self):
        independent_var = []
        dependent_var = []
        for data in self.data:
            dependent_var.append(data.price)
            independent_var.append(data.km)
        independent_mean = np.mean(independent_var)
        dependent_mean = np.mean(dependent_var)
        independent_std_dev = np.std(independent_var)
        dependent_std_dev = np.std(dependent_var)
        for data in self.data:
            data.km = (data.km - independent_mean) / independent_std_dev
        return independent_std_dev, independent_mean

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
    #calculate derivative weight
    def calc_derivative_weight(self, weight, bias):
        error = 0
        for data in self.data:
            predicted_price = self.predict_price(data.km, weight, bias)
            error += data.km * (data.price - predicted_price)
        return (-2 / len(self.data)) * error 

    #calculate derivative bias
    def calc_derivative_bias(self, weight, bias):
        error = 0
        for data in self.data:
            predicted_price = self.predict_price(data.km, weight, bias)
            error += (data.price - predicted_price)
        return (-2 / len(self.data)) * error

    # Calculate the mean square error
    def mse(self, weight, bias):
        mse = 0
        for data in self.data:
            predicted_price = self.predict_price(data.km, weight, bias)
            mse += (predicted_price - data.price) ** 2
        return mse / len(self.data) 

    # train model using learning rate to define calculation rate
    def train_model(self, error_history):
        weight = 0
        bias = 0
        std_deviation, mean = self.extract_data()
        for i in range(70000):
            error_history.append(self.mse(weight, bias))
            weight, bias = self.gradient_descedent(weight, bias, self.learning_rate, error_history)
            self.theta_history.append([weight, bias])
        self.show_plot_data(weight, bias, std_deviation, mean, error_history)
        return weight, bias, std_deviation, mean
    
    def show_plot_data(self, weight, bias, std, mean, mse_history):
        # Extract the km and price data
        Y = [data.price for data in self.data]
        X = [data.km * std + mean for data in self.data]  # Scaled km values
        
        # Predict the prices using the model
        test = [self.predict_price(data.km, weight, bias) for data in self.data]

        # Create a figure with two subplots (1 row, 2 columns)
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

        # First plot: Data points and regression line
        axs[0].scatter(X, Y, color='blue', label='Data points')
        axs[0].plot(X, test, color='red', label='Regression Line')
        axs[0].set_title('Linear Regression - Price vs Scaled Km')
        axs[0].set_xlabel('Scaled Km (km * std + mean)')
        axs[0].set_ylabel('Price')
        axs[0].legend()

        # Second plot: Error history (MSE over iterations)
        axs[1].plot(range(1, len(mse_history) + 1), mse_history, color='red')
        axs[1].set_title('MSE Over Iterations')
        axs[1].set_xlabel('Iteration')
        axs[1].set_ylabel('Mean Squared Error (MSE)')

        # Adjust layout to prevent overlapping subplots
        plt.tight_layout()
        plt.show()

    