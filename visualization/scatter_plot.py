from argparse import ArgumentParser, Namespace
from pandas import DataFrame, read_csv
import seaborn as sns
from matplotlib import pyplot as plt

def _parse_cmd_arguments() -> tuple[DataFrame, Namespace]:
    parser = ArgumentParser(
        prog="scatterplot_plot",
        description="Display scatterplots for comparing two features from a csv file",
    )

    parser.add_argument('file_path',
                        help="path to csv file")
    parser.add_argument('-x', '--x_column', required=True,
                        help="Column for the x-axis in the scatterplot.")
    parser.add_argument('-y', '--y_column', required=True,
                        help="Column for the y-axis in the scatterplot.")
    parser.add_argument('--hue', default=None,
                        help="Name of a categorical column for color coding points.")
    parser.add_argument('-i', '--index', default=False,
                        help="Name of the index column")

    args = parser.parse_args()
    data = read_csv(args.file_path, index_col=args.index)

    return data, args

def plot_scatterplot(data: DataFrame,
                     x_column: str,
                     y_column: str,
                     hue: str | None = None):
    """Plot a scatterplot comparing two columns using seaborn"""

    if x_column not in data.columns:
        print(f"Error: Column '{x_column}' not found in the data.")
        return

    if y_column not in data.columns:
        print(f"Error: Column '{y_column}' not found in the data.")
        return

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x=x_column, y=y_column, hue=hue)
    plt.title(f"Scatterplot of {x_column} vs {y_column}")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend(title=hue)
    plt.show()


def main():
    """Display scatterplots for comparing two features from a csv file"""
    try:
        data, args = _parse_cmd_arguments()
        print("Loaded file", args.file_path)
    except Exception as e:
        print(type(e).__name__, e, sep=": ")
        exit(1)

    sns.set_theme()
    plot_scatterplot(data, args.x_column, args.y_column, args.hue)

if __name__ == "__main__":
    main()
