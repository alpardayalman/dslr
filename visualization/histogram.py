from argparse import ArgumentParser, Namespace
from pandas import DataFrame, read_csv
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.widgets import Button

class HistogramNavigator:
    """Class to handle navigation between histograms."""
    def __init__(self, data: DataFrame, columns: list[str], hue: str | None, bins: int):
        self.data = data
        self.columns = columns
        self.hue = hue
        self.bins = bins
        self.current_index = 0

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.fig.subplots_adjust(bottom=0.2)

        self.axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.btn_prev = Button(self.axprev, 'Previous')
        self.btn_next = Button(self.axnext, 'Next')

        self.btn_prev.on_clicked(self.prev_histogram)
        self.btn_next.on_clicked(self.next_histogram)

        self.plot_histogram()


    def plot_histogram(self):
        """Plot the histogram for the current column."""
        self.ax.clear()
        column = self.columns[self.current_index]
        sns.histplot(data=self.data, x=column, hue=self.hue, bins=self.bins, kde=True, ax=self.ax)
        self.ax.set_title(f"Histogram of {column}")
        self.ax.set_xlabel(column)
        self.ax.set_ylabel("Frequency")
        self.fig.canvas.draw_idle()

    def next_histogram(self, event):
        """Go to the next histogram."""
        self.current_index = (self.current_index + 1) % len(self.columns)
        self.plot_histogram()

    def prev_histogram(self, event):
        """Go to the previous histogram."""
        self.current_index = (self.current_index - 1) % len(self.columns)
        self.plot_histogram()


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


def main():
    """Display interactive histograms from a csv file."""
    try:
        data, args = _parse_cmd_arguments()
        print("Loaded file", args.file_path)
    except Exception as e:
        print(type(e).__name__, e, sep=": ")
        exit(1)

    sns.set_theme()
    navigator = HistogramNavigator(data, args.columns, args.hue, args.bins)
    plt.show()


if __name__ == "__main__":
    main()
