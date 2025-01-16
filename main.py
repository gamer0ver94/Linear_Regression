import matplotlib.pyplot as plt
import car
import argparse
import train_model as train
import numpy as np
import copy
def main():
    # Collect Data from file
    parser = argparse.ArgumentParser(description='Milage :')
    parser.add_argument("milage",
                        help='the mileage to predict price for')
    args = parser.parse_args()
    print(f"Predicting price for mileage: {args.milage}")
    theta = [0,0]
    mileage = float(args.milage)
    
    data = getDataFromFile('./data.csv')
    data = splitData(data)
    normalizedData = normalize_data(data)
    print(data[0].price)
    learningRate = 0.00001
    m = len(data) 
    theta = train.trainModel(theta, m, learningRate, normalizedData)

    # train.mean_square_error(data, theta, m)
    estimated_price = train.estimatePrice(theta, mileage)
    data.append(car.Car(mileage, estimated_price))
    # file = open('./data.csv', 'a')
    # file.write(args.milage + ',' + str(int(estimated_price)) + '\n')
    # file.close
    
    print(f"Estimated Price for {args.milage} mileage: {int(estimated_price)} Euros")
    showGraphData(data, theta)


def normalize_data(data):
    newData = copy.deepcopy(data)
    minPrice = min(data, key=lambda car: car.price).price
    minMilage = min(data, key=lambda car: car.milage).milage
    maxPrice = max(data, key=lambda car:car.price).price
    maxMilage = max(data,key=lambda car:car.milage).milage
    for car in newData:
        car.milage = (car.milage - minMilage) / (maxMilage - minMilage)
        car.price = (car.price - minPrice) / (maxPrice - minPrice)
    return newData

def getDataFromFile(filePath):
    with open(filePath, 'r') as file:
        content = file.read()
    return content


def splitData(data):
    newData = []
    contentLines = data.split('\n')
    for line in contentLines:
        if line == '' or line == 'km,price':
            continue
        splitedLine = line.split(',')
        km = splitedLine[0]
        price = splitedLine[1]
        newData.append(car.Car(float(km), float(price)))
    return newData

def showGraphData(data, theta):
    # Extract mileage (x) and price (y) from the data
    x = [float(car.milage) for car in data]
    y = [float(car.price) for car in data]
    
    # Create the scatter plot for the data points
    plt.scatter(x, y, color='red', label="Data points", alpha=0.7)

    # Generate a smooth regression line based on the model
    x_line = np.linspace(min(x), max(x), 10000)
    y_line = [train.estimatePrice(theta, mileage) for mileage in x_line]
    
    # Plot the regression line
    plt.plot(x_line, y_line, label="Regression Line", color='blue')

    # Adding labels and legend
    plt.xlabel('Mileage (km)')
    plt.ylabel('Price')
    plt.legend()

    # Show the plot
    plt.show()

main()
