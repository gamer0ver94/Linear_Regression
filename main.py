import matplotlib.pyplot as plt
from car import Car
import argparse
from linear_regression import LinearRegression
import numpy as np

def main():
    # Collect Data from file
    parser = argparse.ArgumentParser(description='Milage :')
    parser.add_argument("km",
                        help='the mileage to predict price for')
    args = parser.parse_args()
    theta = [0,0]
    learning_rate = 0.0001
    theta = LinearRegression('data.csv', learning_rate).train_model()
    test = theta[0] + theta[1] * float(args.km)
    print(test)
# def showGraphData(data, theta):
#     # Extract mileage (x) and price (y) from the data
#     x = [float(car.milage) for car in data]
#     y = [float(car.price) for car in data]
    
#     # Create the scatter plot for the data points
#     plt.scatter(x, y, color='red', label="Data points")


#     # Adding labels and legend
#     plt.xlabel('Mileage (km)')
#     plt.ylabel('Price')
#     plt.legend()

#     # Show the plot
#     plt.show()

main()
