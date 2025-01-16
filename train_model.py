import math
def trainModel(theta, m, learningRate, data, max_iterations=1000):
    
    for j in range(max_iterations):
        cost_a = 0
        cost_b = 0
        mse = 0
        for i in range(m):
            predictedValue = estimatePrice(theta, data[i].milage)
            mse = (data[i].price - predictedValue) ** 2
            for h in range(m):
                partialDevone =  - 2 * (data[h].price - (theta[0] + theta[1] * data[h].milage))
                partialDevtwo = (-2 * data[h].milage) * (data[h].price - (theta[0] + (theta[1] * data[h].milage)))
                cost_a = cost_a + partialDevone
                cost_b = cost_b + partialDevtwo
        theta[0] = theta[0] - learningRate * cost_a
        theta[1] = theta[1] - learningRate * cost_b 
        print(str(theta))
    return theta
def estimatePrice(theta, milage):
    return theta[0] + (theta[1] * milage)
