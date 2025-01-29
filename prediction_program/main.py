import argparse
import json
def main():
    #Read the argument
    parser = argparse.ArgumentParser(description='Milage :')
    parser.add_argument("km", nargs='?',
                        help='the mileage to predict price for')
    args = parser.parse_args()
    
    #Message to ask data
    mileage_input = input_validation(args.km)
        
    theta = [0, 0]
    #Open json file to read values from theta
    try:
        with open('../model_data.json') as json_file:
            data = json.load(json_file)
            theta[0] = float(data['theta0'])
            theta[1] = float(data['theta1'])
            mean = float(data['mean'])
            std_dev = float(data['std_deviation'])
        standerized_mileage = (mileage_input - mean) / std_dev
        predicted_price = int(predict_price(theta[0], theta[1], standerized_mileage))
        if predicted_price < 0:
            print('Predicted Velue Pruce is Bellow Zero Each Mean is Not Worth Selling The Car')
        else:
            print('Predicted Price for ' + str(mileage_input) + ' Km is ' + str(predicted_price) + '.')
    except Exception as error:
        print('Sorry, No Data Found or Missing Data For Prediction!\nAs the values theta0 and theta1 were initialized with zero the Value Price Predicted will be 0!')


def predict_price(weight, bias, mileage):
    return bias + (weight * mileage)

def input_validation(input_value):
    value = 0
    if not input_value:
        input_value = input('Please Select The Mileage Your Want To Get A Prediction :')
        if (input_value != ''):
            value = int(input_value)
        else:
            print('Empty Input, Please Insert A Valid Input!')
            exit()
    elif input_value.isnumeric() == False:
        print('Value Not Numeric, Please try a new Input or lower then zero')
        exit()
    else:
        value = int(input_value)
    if value < 0:
        print('There Are No Negative Mileage in a Car. (Wrong Input)')
        exit()
    return value
        
main()