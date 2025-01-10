import matplotlib.pyplot as plt
import car
import argparse
import train_model as train

def main():
    # Collect Data from file
    parser = argparse.ArgumentParser(description='Milage :')
    parser.add_argument("milage",
                        help='the mileage to predict price for')
    args = parser.parse_args()
    print(f"Predicting price for mileage: {args.milage}")
    theta = [0, 0]
    milage = int(args.milage)
    data = getDataFromFile('./data.csv')
    data = splitData(data)
    normalize_mileage(data)
    learningRate = 0.00001
    m = len(data) 
    train.trainModel(theta, m, learningRate, data)
    milage = milage / 1000
    train.mean_square_error(data, theta, m)
    estimated_price = train.estimatePrice(theta, milage)
    file = open('./data.csv', 'a')
    file.write(args.milage + ',' + str(int(estimated_price)) + '\n')
    file.close
    
    print(f"Estimated Price for {args.milage} mileage: {estimated_price}")
    showGraphData(data, theta)



def normalize_mileage(data):
    for car in data:
        car.milage = float(car.milage) / 1000

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
        newData.append(car.Car(km, price))
    return newData

def showGraphData(data, theta):
    x = [float(car.milage) for car in data]
    y = [float(car.price) for car in data]
    plt.plot(x, y, 'ro', label="Data points")
    x_line = range(int(min(x)), int(max(x)) + 1)
    y_line = [train.estimatePrice(theta, mileage) for mileage in x_line]
    plt.plot(x_line, y_line, label="Regression Line")
    plt.xlabel('Mileage (km)')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

main()
