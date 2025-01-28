
from utils.linear_regression import LinearRegressions

def main():
    error_history = []
    # Collect Data from file
    weight, bias = LinearRegressions('data.csv', learning_rate = 0.0001).train_model(error_history)
main()