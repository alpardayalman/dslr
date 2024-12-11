from argparse import ArgumentParser
import pandas as pd
import numpy as np

from regression.regressor import OneVsAllRegressor


def _parse_cmd_arguments():
    parser = ArgumentParser(
        prog="logreg train",
        description="One vs All Logistic Regression"
    )

    parser.add_argument('file_path',
                        help="path to csv file")
    parser.add_argument('-l', '--label', required=True,
                        help="name of the label column")
    parser.add_argument('-i', '--index', default=False,
                        help="name of the index column")
    parser.add_argument('-a', '--alpha', default=0.1,
                        help="learning rate")
    parser.add_argument('--itrmax', default=1000,
                        help="maximum number of iterations in algorithm")
    parser.add_argument('-m', '--method', default="gradient",
                        choices=["gradient", "sgradient"])

    args = parser.parse_args()
    
    data = pd.read_csv(args.file_path, index_col=args.index)

    y = data[args.label]
    X = data.drop(columns=[args.label])
    X = X.select_dtypes(include=['number'])

    return X, y, args

def min_max_normalize(X, feature_range=(0, 1)):
    """
        min max normalizer klasik 
    """
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X_std = (X - X_min) / (X_max - X_min)
    X_scaled = X_std * (feature_range[1] - feature_range[0]) + feature_range[0]
    return X_scaled, X_min, X_max

def main():
    """Perform logistic regression on data from a csv file"""
    try:
        X, y, args = _parse_cmd_arguments()
        print("Loaded file", args.file_path)
    except Exception as e:
        print(type(e).__name__, e, sep=": ")
        exit(1)

    X = X.replace(np.nan, 0)
    X, _, _ = min_max_normalize(X)
    
    reg = OneVsAllRegressor()
    reg.fit(X, y, alpha=args.alpha, max_itr=args.itrmax)

    classes = reg.transform_info["classes"]
    weights = pd.DataFrame(reg.weights, columns=pd.Index(classes))

    print(weights)

    name = reg.transform_info["name"]
    weights.to_csv(name + "_weights.csv")


if __name__ == "__main__":
    main()
