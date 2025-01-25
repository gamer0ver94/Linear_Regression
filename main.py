
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
    learning_rate = 0.000001
    weight, bias = LinearRegressions('data.csv', learning_rate).train_model(error_history)
    #standarize value
    stdv = (float(args.km) - 6331.833333333333) / 1319.6541927360354
    print('std mile : ' + str(stdv))
    test = bias + (weight * stdv)
    test = (test * 1319.6541927360354) + 6331.833333333333
    print('Price', int(test))
    print('mse = ' + str(np.mean(error_history)))
    beta_test(args.km)

def beta_test(km):
    file_data = pd.read_csv('data.csv')
    x = []
    y = []
    for i in range(len(file_data)):
        x.append(file_data.km[i])
        y.append(file_data.price[i])
    x = np.array(x).reshape(-1, 1)
    
    reg = LinearRegression().fit(x,y)
    print('Value predicted by real model for ' + str(km) +' km is ' + str(reg.predict(np.array([[int(km)]]))))

main()