import argparse
import json
def main():
    #Read the argument
    parser = argparse.ArgumentParser(description='Milage :')
    parser.add_argument("km", nargs='?',
                        help='the mileage to predict price for')
    args = parser.parse_args()
    if not args.km:
        #Message to ask data
        mileage_input = float(input('Please Select The Mileage Your Want To Get A Prediction :'))
    else:
        mileage_input = int(args.km)
    if mileage_input < 0:
        print('There Are No Negative Mileage in a Car. (Wrong Input)')
        exit()
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
            print('Predicted Price for ' + str(mileage_input) + ' Km is ' + str(predicted_price) + ' Euros.')
    except Exception as error:
        print('Sorry, No Data Found For Prediction! or ' + str(error))


def predict_price(weight, bias, mileage):
    return bias + (weight * mileage)


main()