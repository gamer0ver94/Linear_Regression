
from car import Car
import argparse
from linear_regressions import LinearRegressions
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
def main():
    error_history = []
    # Collect Data from file
    parser = argparse.ArgumentParser(description='Milage :')
    parser.add_argument("km",
                        help='the mileage to predict price for')
    args = parser.parse_args()
    theta = [0,0]
    learning_rate = 0.00001
    # theta = LinearRegression('data.csv', learning_rate).train_model(error_history)
    # #standarize value
    # stdv = (int(args.km) - 6331.833333333333) / 1319.6541927360354
    # print('std mile : ' + str(stdv))
    # test = theta[0] + (theta[1] * stdv)
    # test = (test * 1319.6541927360354) + 6331.833333333333
    # print('Price', int(test))
    # print('mse = ' + str(np.mean(error_history)))
    # for data in error_history:
    #     print(data)
    file_data = pd.read_csv('data.csv')
    x = []
    y = []
    for data in file_data:
        x.append(data.km)
        y.append(data.price)
    reg = LinearRegression().fit(x,y)
    print(reg.score)

main()
