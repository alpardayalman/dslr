import pandas as pd
from .normalization import minmax_scale


def fillna_constant_minmax_scale(data: pd.DataFrame,
                                 constant=0,
                                 feature_range=(0, 1)):
    """Replace NA with a constant value and apply min max scaling on data"""
    data = data.fillna(constant)
    data = minmax_scale(data, feature_range)
    return data
