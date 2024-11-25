from argparse import ArgumentParser, Namespace
from pandas import DataFrame, read_csv
import seaborn as sns
from matplotlib import pyplot as plt

def _parse_cmd_arguments() -> tuple[DataFrame, Namespace]:
    parser = ArgumentParser(
        prog="histogram_plot",
        description="Display histograms from a csv file",
    )

    parser.add_argument('file_path',
                        help="path to csv file")
    parser.add_argument('-c', '--columns', nargs='+', required=True,
                        help="Columns to plot histograms for. Ex: -c col3 col5, use '*' to view all columns")
    parser.add_argument('--hue', default=None,
                        help="Name of a categorical column for color")
    parser.add_argument('-i', '--index', default=False,
                        help="Name of the index column")
    parser.add_argument('--bins', type=int, default=20,
                        help="Number of bins for the histogram (default: 20)")

    args = parser.parse_args()
    print(args.index)
    data = read_csv(args.file_path, index_col=args.index)

    if "*" in args.columns:
        args.columns = data.select_dtypes(include=['number']).columns.to_list()

    return data, args

def plot_histograms(data: DataFrame,
                    columns: list[str],
                    hue: str | None = None,
                    bins: int = 20):
    """Plot histograms for the selected columns using seaborn"""

    for i, column in enumerate(columns):
        if column not in data.columns:
            print(f"Warning: Column '{column}' not found in the data.")
            continue
        
        plt.figure(figsize=(8, 6))
        sns.histplot(data[column], kde=True, hue=hue, bins=bins)
        plt.title(f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()

def main():
    """Display histograms from a csv file"""
    try:
        data, args = _parse_cmd_arguments()
        print("Loaded file", args.file_path)
    except Exception as e:
        print(type(e).__name__, e, sep=": ")
        exit(1)

    sns.set_theme()
    plot_histograms(data, args.columns, args.hue, args.bins)

if __name__ == "__main__":
    main()
