import argparse

def main():
    parser = argparse.ArgumentParser(description='Milage :')
    parser.add_argument("km",
                        help='the mileage to predict price for')
    args = parser.parse_args()
    km = args.km