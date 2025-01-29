import json
import pandas as pd
import numpy as np
def main():
    with open('../model_data.json') as json_file:
        json_data = json.load(json_file)
    real_data = pd.read_csv('../training_model/data_sets/data.csv')
    std_derivation = float(json_data['std_deviation'])
    mean = float(json_data['mean'])
    weight = float(json_data['theta0'])
    bias = float(json_data['theta1'])
    Y = real_data.price.copy()
    X = (real_data.km.copy() - mean) / std_derivation
    # for data in X:
    #     data = (data - mean) / std_derivation
    mserror = r_squared(X,Y, weight, bias)
    print(str(int(mserror * 100)) + '%')
    #   0 - 100%
    #   er- 
    # print(str(100 - percentage))
    return

def r_squared(X, Y, weight, bias):
    total_sum_of_squares = 0
    residual_sum_of_squares = 0
    mean_y = np.mean(Y)
    
    for i in range(len(X)):
        predicted_value = predict_price(weight, bias, X[i])
        residual_sum_of_squares += (Y[i] - predicted_value) ** 2
        total_sum_of_squares += (Y[i] - mean_y) ** 2
    
    r_squared_value = 1 - (residual_sum_of_squares / total_sum_of_squares)
    return r_squared_value

def predict_price(weight, bias, target):
    return bias + (weight * target)
main()

