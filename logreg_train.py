from argparse import ArgumentParser
import pandas as pd
from sklearn.model_selection import train_test_split

from regression.regressor import OneVsAllRegressor
from preprocessing.procedures import fillna_trim_minmax_scale

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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
    parser.add_argument('-a', '--alpha', default=0.1, type=float,
                        help="learning rate")
    parser.add_argument('--itrmax', default=1000, type=int,
                        help="maximum number of iterations in algorithm")
    parser.add_argument('-m', '--method', default="gradient",
                        choices=["gradient", "sgradient"])
    parser.add_argument('-f', '--FullTrain', action='store_true', 
                        help="Enable FullTrain mode.")

    args = parser.parse_args()

    data = pd.read_csv(args.file_path, index_col=args.index)

    y = data[args.label]
    # X = data.drop(columns=[args.label]) I need to remouve this sorry <- preprocessing purposes
    X = data
    # X = X.select_dtypes(include=['number'])

    return X, y, args


def main():
    """Perform logistic regression on data from a csv file"""
    try:
        X, y, args = _parse_cmd_arguments()
        print("Loaded file", args.file_path)
    except Exception as e:
        print(type(e).__name__, e, sep=": ")
        exit(1)

    # Preprocessing
    X = fillna_trim_minmax_scale(X, label=args.label)

    # Train test split
    if args.FullTrain:
        X_train = X
        y_train = y
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    # Regression
    reg = OneVsAllRegressor()
    reg.fit(X_train, y_train, alpha=args.alpha, max_itr=args.itrmax)

    classes = reg.transform_info["classes"]
    weights = pd.DataFrame(reg.weights, columns=pd.Index(classes))

    print("=" * 20)
    print(weights.head())
    print("=" * 20)

    if not args.FullTrain:
        print("Model accuracy", reg.score(X_test, y_test))

    path = "weights.csv"
    weights.to_csv(path)

    print("Weights are written into file", "\"" + path + "\"")


if __name__ == "__main__":
    main()
