
from utils.linear_regression import LinearRegressions
import json
def main():
    error_history = []
    # Collect Data from file
    weight, bias, std_deviation, mean = LinearRegressions('data_sets/data.csv', learning_rate = 0.0001).train_model(error_history)
    json_obj = {
        'theta0':str(weight),
        'theta1':str(bias),
        'std_deviation':str(std_deviation),
        'mean':str(mean),
    }
    with open('../model_data.json', 'w') as json_file:
        json.dump(json_obj, json_file)
    print(weight, bias)
main()