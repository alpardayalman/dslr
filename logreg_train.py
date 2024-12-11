from argparse import ArgumentParser
import pandas as pd

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

    return X, y, args


def main():
    """Perform logistic regression on data from a csv file"""
    try:
        X, y, args = _parse_cmd_arguments()
        print("Loaded file", args.file_path)
    except Exception as e:
        print(type(e).__name__, e, sep=": ")
        exit(1)

    # apply data transformation on X...

    reg = OneVsAllRegressor()
    reg.fit(X, y, alpha=args.alpha, max_itr=args.itrmax)

    classes = reg.transform_info["classes"]
    weights = pd.DataFrame(reg.weights, columns=pd.Index([classes]))

    weights.to_csv("weights.csv")


if __name__ == "__main__":
    main()
