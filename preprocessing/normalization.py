import pandas as pd


def minmax_scale(data: pd.DataFrame, feature_range=(0, 1)):
    min = data.min(axis=0)
    max = data.max(axis=0)
    scl = (data - min) / (max - min)

    if feature_range != (0, 1):
        fmin = feature_range[0]
        fmax = feature_range[1]
        scl = scl * (fmax - fmin) + fmin

    return scl
