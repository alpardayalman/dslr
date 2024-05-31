import pandas as pd
import numpy as np
import sys


def my_describe(data):
    """
    This function will print the describe of the data
    :param data: path to the data
    :return: None
    """
    mydata = pd.read_csv(data)
    num = mydata.select_dtypes(include=np.number)
    print(num.columns)
    mynewdata = pd.DataFrame(columns=num.columns, index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
    for i in num.columns:
        mynewdata[i] = [num[i].count() ,num[i].mean(), num[i].std(), num[i].min(), num[i].quantile(0.25),num[i].quantile(0.50),num[i].quantile(0.75), num[i].max()]
    print(mynewdata)
    # print(mydata.describe())
    # print(num.describe())
    return mynewdata

def main():

    if len(sys.argv) != 2:
        print('describe.py <dataset>.csv')
        return

    if sys.argv[1] == 'help':
        print('This script will print the describe of the data')
        print('Usage: python describe.py <dataset>.csv ')
        return
    my_describe(sys.argv[1])

if __name__ == '__main__':
    main()