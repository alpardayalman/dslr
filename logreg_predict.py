from argparse import ArgumentParser
import pandas as pd

from regression.regressor import OneVsAllRegressor
from preprocessing.procedures import fillna_constant_minmax_scale

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def _parse_cmd_arguments():
    parser = ArgumentParser(
        prog="logreg predict",
        description="Predict via One vs All Logistic Regression"
    )

    parser.add_argument('data_path',
                        help="path to data csv file")
    parser.add_argument('weights_path',
                        help="path to weights csv file")
    parser.add_argument('-l', '--label', required=True,
                        help="name of the label")
    parser.add_argument('-i', '--index', default=False,
                        help="name of the index column in data file")
    parser.add_argument('-o', '--output', required=True,
                        help="name of the output file")
    parser.add_argument('-p', '--probability', action="store_true",
                        help="write probability distribution to output file")

    args = parser.parse_args()
    data = pd.read_csv(args.data_path, index_col=args.index)

    if args.label in data.columns:
        X = data.drop(columns=[args.label])
    else:
        X = data
    X = X.select_dtypes(include=['number'])

    LogRegressor = OneVsAllRegressor()
    LogRegressor.load_weights(args.weights_path, args.label)

    return X, LogRegressor, args


def main():
    """Perform logistic regression on data from a csv file"""
    try:
        X, regressor, args = _parse_cmd_arguments()
        print("Loaded file", args.data_path)
        print("Loaded weights")
    except Exception as e:
        print(type(e).__name__, e, sep=": ")
        exit(1)

    # Preprocessing
    X = fillna_constant_minmax_scale(X)

    if args.probability:
        pred = regressor.predict_tabular(X)
    else:
        pred = regressor.predict_class(X)
        pred = pd.DataFrame(pred).set_index(X.index)

    path = args.output + ".csv"
    pred.to_csv(path)

    print("Results are written into file", "\"" + path + "\"")


if __name__ == "__main__":
    main()
