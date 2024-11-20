from argparse import ArgumentParser, Namespace
from pandas import DataFrame, read_csv
import seaborn as sns
from matplotlib import pyplot as plt

def _parse_cmd_arguments() -> tuple[DataFrame, Namespace]:
    parser = ArgumentParser(
        prog="pair_plot",
        description="display a pair-plot from a csv file",
    )

    parser.add_argument('file_path',
                        help="path to csv file")
    parser.add_argument('-c', '--columns', nargs='+', required=True,
                        help="Ex: -c col4 col2 col3, use '*' to view all columns")
    parser.add_argument('--hue', default=None,
                        help="name of a categorical column for color")
    parser.add_argument('-i', '--index', default=False,
                        help="name of the index column")

    args = parser.parse_args()
    data = read_csv(args.file_path, index_col=args.index)

    if "*" in args.columns:
        args.columns = data.select_dtypes(include=['number']).columns.to_list()
    
    return data, args

    

def pair_plot(data: DataFrame,
              features: list[str],
              hue: str | None = None):
    """Set up a pair-plot in seaborn"""
    sns.pairplot(data, vars=features, hue=hue)

def main():
    """Display a pair-plot from a csv file"""
    try:
        data, args = _parse_cmd_arguments()
        print("Loaded file", args.file_path)
    except Exception as e:
        print(type(e).__name__, e, sep=": ")
        exit(1)

    sns.set_theme()
    pair_plot(data, args.columns, args.hue)
    plt.show()

if __name__ == "__main__":
    main()
