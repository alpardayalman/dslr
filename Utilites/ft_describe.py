import pandas as pd
import numpy as np


def ft_describe(data, debug=False):
    """
    This function will print the describe of the data
    :param data: path to the data
    :return: None
    """
    mydata = pd.read_csv(data)
    num = mydata.select_dtypes(include=np.number)
    mynewdata = pd.DataFrame(columns=num.columns, index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
    for i in num.columns:
        mynewdata[i] = [count_data(num[i]) ,mean_data(num[i]), std_data(num[i]), min_data(num[i]), quantile_data(num[i], 0.25), quantile_data(num[i], 0.50), quantile_data(num[i], 0.75), max_data(num[i])]
    if debug:
        print(mydata.describe())
        print("-"*50)
    return mynewdata

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

def std_data(num):
    i = 0
    mean = mean_data(num)
    for j in num:
        if np.isnan(j):
            continue
        i += (j - mean)**2
    return (i/(count_data(num)-1))**0.5

def min_data(num):
    num = num.sort_values()
    return num.iloc[0]

def quantile_data(num, q):
    num = num.sort_values()
    n = count_data(num)
    rank = (n-1)*q
    lower_index = int(rank)
    upper_index = lower_index + 1
    
    if upper_index >= n:
        return num.iloc[lower_index]
    lower_value = num.iloc[lower_index]
    upper_value = num.iloc[upper_index]
    weight = rank - lower_index
    return lower_value * (1 - weight) + upper_value * weight

def max_data(num):
    num = num.sort_values()
    while np.isnan(num.iloc[-1]):
        num = num.drop(num.index[-1])
    return num.iloc[-1]