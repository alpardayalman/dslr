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

def mean_data(num):
    i = 0
    for j in num:
        if np.isnan(j):
            continue
        i += j
    return i/count_data(num)

def fill_nan_with_trimmed_mean(df, class_col, trim_frac=0.05):

    feature_cols = df.select_dtypes(include=['number']).columns
    for feature in feature_cols:
        grouped = df.groupby(class_col)
        for house, group in grouped:
            values = group[feature].dropna().values
            
            sorted_values = np.sort(values)
            
            trim_count = int(len(sorted_values) * trim_frac)
            
            if len(sorted_values) > 2 * trim_count:
                trimmed_values = sorted_values[trim_count:-trim_count]
            else:
                trimmed_values = sorted_values
                
            trimmed_mean = mean_data(trimmed_values) if len(trimmed_values) > 0 else np.nan
            
            df.loc[(df[class_col] == house) & (df[feature].isna()), feature] = trimmed_mean
    return df

def fillna_constant_minmax_scale(data: pd.DataFrame,
                                 train=True,
                                 label='Hogwarts House',
                                 constant=0,
                                 feature_range=(0, 1)):
    """Replace NA with a constant value and apply min max scaling on data"""

    if train:
        data = fill_nan_with_trimmed_mean(data, label)
    if label in data.columns:
        data = data.drop(columns=[label])
    data = data.select_dtypes(include=['number'])
    data = minmax_scale(data, feature_range)
    print(data.isna().sum())
    return data
