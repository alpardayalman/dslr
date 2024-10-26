import sys
from Utilites import ft_describe
import pandas as pd


def main():

    if len(sys.argv) != 2:
        print('describe.py <dataset>.csv')
        return

    if sys.argv[1] == 'help':
        print('This script will print the describe of the data')
        print('Usage: python describe.py <dataset>.csv ')
        return
    print(ft_describe(sys.argv[1], debug=True))
    
    print(pd.read_csv(sys.argv[1]).describe())


if __name__ == '__main__':
    main()
