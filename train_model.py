import math
def trainModel(theta, m, learningRate, data):
    for i in range(1000):
            theta[0] = calculate_intercept(learningRate, m, data, theta)
            theta[1] = calculate_slope(learningRate, m, data, theta)

def calculate_slope(learning_rate, m, data, theta):
    error = 0
    for i in range(m):
        predicted_price = estimatePrice(theta, float(data[i].milage))
        error += (predicted_price - float(data[i].price)) * float(data[i].milage)
    return theta[1] - learning_rate * (error / m)

def calculate_intercept(learning_rate, m, data, theta):
    error = 0
    for i in range(m):
        predicted_price = estimatePrice(theta, float(data[i].milage))
        error += predicted_price - float(data[i].price)
    return theta[0] - learning_rate * (error / m)

def estimatePrice(theta, milage):
    return theta[0] + (theta[1] * milage)

import math

def mean_square_error(data, theta, m):
    predictData = []
    mean_square_values = 0
    for item in data:
        # Assuming 'estimatePrice' function predicts the value given theta and milage
        predicted_price = estimatePrice(theta, float(item.milage))
        predictData.append(predicted_price)

    # Calculate the squared errors
    for i in range(m):
        error = float(data[i].milage) - predictData[i]
        mean_square_values += error ** 2  # Square the error

    # Compute the mean of the squared errors
    mse = mean_square_values / m
    print("Mean Squared Error:", mse)

    # If you want RMSE (Root Mean Squared Error), use the following:
    rmse = math.sqrt(mse)
    print("Root Mean Squared Error:", rmse)

