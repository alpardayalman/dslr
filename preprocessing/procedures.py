import pandas as pd
from .normalization import minmax_scale
import numpy as np

def count_data(num):
    try:
        num = num.astype('float')
        num = num[~np.isnan(num)]
        return len(num)
    except:
        return len(num)

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

def compute_median_using_quantile(column):
    if column.isnull().all():
        return np.nan
    return quantile_data(column.dropna(), 0.5)

def fillna_constant_minmax_scale(data: pd.DataFrame,
                                 feature_range=(0, 1)):
    """Replace NA with median using quantile_data and apply min-max scaling"""
    medians = data.apply(compute_median_using_quantile)
    data = data.fillna(medians)
    data = minmax_scale(data, feature_range)
    return data
