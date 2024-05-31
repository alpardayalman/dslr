import pandas as pd
import numpy as np
import sys

def count_data(num):
    try:
        num = num.astype('float')
        num = num[~np.isnan(num)]
        return len(num)
    except:
        return len(num)

def mean_data(num):
    i = 0
    for j in num:
        if np.isnan(j):
            continue
        i += j
    return i/count_data(num)

def my_describe(data):
    """
    This function will print the describe of the data
    :param data: path to the data
    :return: None
    """
    mydata = pd.read_csv(data)
    num = mydata.select_dtypes(include=np.number)
    mynewdata = pd.DataFrame(columns=num.columns, index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
    for i in num.columns:
        mynewdata[i] = [count_data(num[i]) ,mean_data(num[i]), num[i].std(), num[i].min(), num[i].quantile(0.25),num[i].quantile(0.50),num[i].quantile(0.75), num[i].max()]
    print(mynewdata)
    print("------------------------------------\n")
    print(mydata.describe())
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