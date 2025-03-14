import pandas as pd
import numpy as np
import logging
import colorlog

import plotly.graph_objects as go
import plotly.figure_factory as ff

from typing import Tuple, Optional, List
from functools import wraps
from os.path import exists

# Monitoring
import time
import tracemalloc
import psutil
import cProfile
import pstats
import io


def init_logger():
    """Initializes the logger with a custom format."""
    log_format = "%(log_color)s%(levelname)s%(reset)s: %(asctime)s.%(msecs)03d - %(message)s%(reset)s"
    color_formatter = colorlog.ColoredFormatter(
        log_format,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(color_formatter)
    logging.basicConfig(level=logging.INFO, handlers=[console_handler])


logger = logging.getLogger(__name__)
init_logger()

# Constants for colors
LINE_COLORS = ["yellow", "green", "red", "blue"]
ARROW_COLORS = ["black", "black", "black", "black"]


class DataLoader:
    """Handles loading data from csv file."""

    @staticmethod
    def load_data(
        file_path: str, rows: int = 0, usecols: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Loads data from a CSV file into a pandas DataFrame.

        Args:
            file_path: Path to the CSV file
            rows: Number of rows to read (0 for all rows)
            usecols: List of specific columns to load (will automatically include Date and Time)

        Returns:
            DataFrame containing the loaded data or None if loading fails
        """
        if not exists(file_path):
            logger.error("File not found: %s", file_path)
            return None

        try:
            # Ensure Date and Time columns are included for plotting
            if usecols is not None:
                usecols = list(set(usecols) | {"Date", "Time"})

            df = pd.read_csv(
                file_path,
                skiprows=2,  # Skip header rows
                nrows=rows if rows else None,
                low_memory=True,
                usecols=usecols,
                parse_dates=[[0, 1]],  # Combine Date and Time columns
            )

            if df is None or df.empty:
                logger.error("Empty DataFrame loaded")
                return None

            # Clean up any unnamed columns
            df.drop(
                columns=[col for col in df.columns if col.startswith("Unnamed")],
                inplace=True,
                errors="ignore",
            )

            logger.info("Data loaded successfully: %s rows", len(df))
            return df

        except Exception as e:
            logger.error("Failed to load data: %s", e)
            return None


class DataProcessor:
    """Handles data preprocessing and transformations."""

    @staticmethod
    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the data by:
        - taking the absolute values of the power columns.

        Args:
            df: Input DataFrame with power data

        Returns:
            Preprocessed DataFrame with absolute power values
        """
        if df is None or df.empty:
            logger.error("Empty DataFrame provided for preprocessing")
            return pd.DataFrame()

        try:
            # Power columns should be absolute values
            power_cols = ["PA", "PB", "PC"]
            df[power_cols] = df[power_cols].abs()

            logger.info("Data preprocessing completed")
            return df

        except KeyError as e:
            logger.error("Preprocessing failed due to missing columns: %s", e)
            return pd.DataFrame()


class Plotter:
    """Handles visualization of power data."""

    def __init__(self):
        """Initialize the Plotter with an empty figure."""
        self.fig = None

    def plot_time_series(
        self,
        df: pd.DataFrame,
        columns: List[str],
        title: str,
        min_max_arrows: bool = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        downsample: int = 1,
    ) -> Optional[go.Figure]:
        """
        Plot time series data with optional min/max annotations and date filtering.

        Args:
            df: DataFrame containing time series data with Date_Time column
            columns: List of column names to plot
            title: Chart title
            min_max_arrows: Whether to add min/max annotations
            start_date: Optional start date filter (format: 'YYYY-MM-DD HH:MM')
            end_date: Optional end date filter (format: 'YYYY-MM-DD HH:MM')
            downsample: Factor to downsample data for better performance (1 = no downsampling)

        Returns:
            Plotly figure object or None if plotting fails
        """
        if df.empty:
            logger.error("Empty DataFrame provided for time series plot")
            return None

        # Apply date filtering if specified
        if start_date or end_date:
            df = self._filter_date_range(df, start_date, end_date)

        # Fill gaps in time series data
        df = self._fill_missing_time_with_zeros(df, columns)

        # Downsample data if needed for performance
        if downsample > 1:
            df = df.iloc[::downsample].copy()

        self.fig = go.Figure()

        # Add each column as a separate trace
        for idx, col in enumerate(columns):
            self.fig.add_trace(
                go.Scatter(
                    x=df["Date_Time"],
                    y=df[col],
                    mode="lines",
                    name=col,
                    line=dict(color=LINE_COLORS[idx % len(LINE_COLORS)]),
                )
            )

        self.fig.update_layout(
            title=title,
            xaxis_title="Date and Time",
            yaxis_title="Values",
            hovermode="closest",
            title_font=dict(size=16, color="black"),
            xaxis=dict(title_font=dict(size=14), tickfont=dict(size=12)),
            yaxis=dict(title_font=dict(size=14), tickfont=dict(size=12)),
            legend=dict(font=dict(size=12)),
        )

        if min_max_arrows:
            self._add_min_max_arrows(df, columns)

        self.fig.show()
        return self.fig

    def plot_gaussian_distribution_v1(
        self,
        df: pd.DataFrame,
        columns: list[str],
        title: str,
        start_date: str = None,
        end_date: str = None,
    ):
        """
        Plot a Gaussian distribution chart using ff.create_distplot

        Args:
            df: DataFrame containing data
            columns: List of column names to plot distributions for
            title: Chart title
            start_date: Optional start date filter (format: 'YYYY-MM-DD HH:MM')
            end_date: Optional end date filter (format: 'YYYY-MM-DD HH:MM')

        Returns:
            Plotly figure object or None if plotting fails
        """
        if df is None or df.empty:
            logger.error("Empty DataFrame provided for Gaussian distribution plot.")
            return None

        # hist_data = [df[col] for col in columns]
        hist_data = [df[col][df[col] != 0] for col in columns]
        group_labels = columns

        df = self._filter_date_range(df, start_date, end_date)

        self.fig = ff.create_distplot(hist_data, group_labels, bin_size=1)

        self.fig.update_layout(
            title=title,
            xaxis_title="Value",
            yaxis_title="Density",
            title_font=dict(size=16, color="black"),
            xaxis=dict(title_font=dict(size=14), tickfont=dict(size=12)),
            yaxis=dict(title_font=dict(size=14), tickfont=dict(size=12)),
            legend=dict(title_text="Columns", font=dict(size=12)),
        )

        self.fig.show()
        return self.fig  # Return figure instead of showing it

    def plot_gaussian_distribution_v2(
        self,
        df: pd.DataFrame,
        columns: List[str],
        title: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        bins: int = 50,
    ) -> Optional[go.Figure]:
        """
        Plot a Gaussian distribution chart using histogram approach.

        Args:
            df: DataFrame containing data
            columns: List of column names to plot distributions for
            title: Chart title
            start_date: Optional start date filter (format: 'YYYY-MM-DD HH:MM')
            end_date: Optional end date filter (format: 'YYYY-MM-DD HH:MM')
            bins: Number of histogram bins

        Returns:
            Plotly figure object or None if plotting fails
        """
        if df is None or df.empty:
            logger.error("Empty DataFrame provided for Gaussian distribution plot")
            return None

        # Filter by date range if specified
        df = self._filter_date_range(df, start_date, end_date)
        self.fig = go.Figure()

        # Create a distribution plot for each column
        for idx, col in enumerate(columns):
            # Exclude zeros for better distribution visualization
            data = df[col][df[col] != 0].dropna()
            if data.empty:
                logger.warning("No valid data found for column: %s", col)
                continue

            # Compute a histogram with density normalization
            hist, bin_edges = np.histogram(data, bins=bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            self.fig.add_trace(
                go.Scatter(
                    x=bin_centers,
                    y=hist,
                    mode="lines",
                    name=col,
                    line=dict(color=LINE_COLORS[idx % len(LINE_COLORS)]),
                )
            )

        # Configure layout
        self.fig.update_layout(
            title=title,
            xaxis_title="Value",
            yaxis_title="Density",
            title_font=dict(size=16, color="black"),
            xaxis=dict(title_font=dict(size=14), tickfont=dict(size=12)),
            yaxis=dict(title_font=dict(size=14), tickfont=dict(size=12)),
            legend=dict(title_text="Columns", font=dict(size=12)),
        )

        self.fig.show()
        return self.fig

    def _filter_date_range(
        self,
        df: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Filter DataFrame between start_date and end_date based on Date_Time column.

        Args:
            df: DataFrame with Date_Time column
            start_date: Start date string (format: 'YYYY-MM-DD HH:MM')
            end_date: End date string (format: 'YYYY-MM-DD HH:MM')

        Returns:
            Filtered DataFrame
        """
        if df is None or df.empty:
            logger.error("Empty DataFrame provided to filter date range")
            return pd.DataFrame()

        df_copy = df.copy()
        try:
            if start_date:
                start_date = pd.to_datetime(start_date, errors="coerce")
                df_copy = df_copy[df_copy["Date_Time"] >= start_date]
            if end_date:
                end_date = pd.to_datetime(end_date, errors="coerce")
                df_copy = df_copy[df_copy["Date_Time"] <= end_date]
        except Exception as e:
            logger.error("Failed to filter date range: %s", e)
            return df
        return df_copy

    def _fill_missing_time_with_zeros(
        self, df: pd.DataFrame, columns: List[str]
    ) -> pd.DataFrame:
        """
        Fill gaps in time series data by inserting rows with zeros.

        This method detects gaps in the Date_Time series (greater than 2 hours)
        and fills them by inserting rows with zeros in the specified columns.

        Args:
            df: DataFrame with Date_Time column
            columns: List of columns to fill with zeros

        Returns:
            DataFrame with gaps filled
        """
        if "Date_Time" not in df.columns:
            logger.error("DataFrame does not contain 'Date_Time' column")
            return df

        # Ensure Date_Time is datetime type
        if not pd.api.types.is_datetime64_any_dtype(df["Date_Time"]):
            df["Date_Time"] = pd.to_datetime(df["Date_Time"], errors="coerce")

        # Sort by time and calculate time differences
        df = df.sort_values("Date_Time").reset_index(drop=True)
        df["Time_Diff"] = df["Date_Time"].diff()

        # Identify gaps exceeding 2 hours
        gap_indices = df.index[df["Time_Diff"] > pd.Timedelta(hours=2)]
        new_rows = []

        # Fill each gap with 2-hour interval rows
        for i in gap_indices:
            start_time = df.loc[i - 1, "Date_Time"]
            end_time = df.loc[i, "Date_Time"]
            # Generate timestamps at 2-hour intervals (excluding endpoints)
            missing_times = pd.date_range(
                start=start_time + pd.Timedelta(hours=2),
                end=end_time - pd.Timedelta(hours=2),
                freq="2H",
            )

            # Create rows with zeros for each missing timestamp
            for t in missing_times:
                row = {col: 0 for col in columns}
                row["Date_Time"] = t
                new_rows.append(row)

        # Append new rows and clean up
        if new_rows:
            zero_fill_df = pd.DataFrame(new_rows)
            df = pd.concat([df, zero_fill_df], ignore_index=True)

        df.drop(columns=["Time_Diff"], inplace=True)
        df = (
            df.drop_duplicates(subset="Date_Time")
            .sort_values("Date_Time")
            .reset_index(drop=True)
        )
        return df

    def _add_min_max_arrows(self, df: pd.DataFrame, columns: List[str]) -> None:
        """
        Add min and max value annotations to the plot for each column.

        Args:
            df: DataFrame with data
            columns: List of columns to annotate
        """
        for idx, col in enumerate(columns):
            try:
                # Create a mask to filter out rows where the specified column value is less than or equal to 0
                mask = df[col] > 0.01

                # Find max and min values
                if mask.any():
                    max_row = df[mask].loc[df[mask][col].idxmax()]
                    min_row = df[mask].loc[df[mask][col].idxmin()]
                    max_value = max_row[col]
                    min_value = min_row[col]

                    # Add annotations
                    self._add_annotation(max_row, col, idx, max_value, "Max")
                    self._add_annotation(min_row, col, idx, min_value, "Min")
                else:
                    logger.warning("No valid data found for %s greater than 0.01", col)
            except ValueError as e:
                logger.warning("Error finding max/min for %s: %s", col, e)

    def _add_annotation(
        self, row: pd.Series, col: str, idx: int, value: float, label: str
    ) -> None:
        """
        Add a formatted annotation to the plot.
        
        Args:
            row: DataFrame row containing the point to annotate
            col: Column name
            idx: Index for color selection
            value: Value to display
            label: Label text (e.g., "Max" or "Min")
        """
        self.fig.add_annotation(
            text=f"<b>{label}:</b><br>{col} = {value:.2f}",
            x=row["Date_Time"],
            y=row[col],
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor=ARROW_COLORS[idx % len(ARROW_COLORS)],
            ax=0,
            ay=-40,
            font=dict(size=12, color="black"),
            bgcolor="rgba(255, 255, 255, 0.66)",
            bordercolor="black",
            borderwidth=1,
            borderpad=3,
        )


class DescriptiveStats:
    """Handles statistical analysis of power data."""
    
    @staticmethod
    def describe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Calculate descriptive statistics for power data.
        
        Args:
            df: DataFrame containing power data
            
        Returns:
            Tuple containing:
            - DataFrame of maximum values by category
            - DataFrame of minimum values by category
            - DataFrame of energy changes (first, last, and difference)
        """
        if df.empty:
            logger.error("Empty DataFrame provided for descriptive statistics")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        max_dfs = []
        min_dfs = []

        # Define sections for metrics grouping
        sections = {
            "Current": ["IA", "IB", "IC"],
            "Voltage": ["UA", "UB", "UC"],
            "Power": ["PA", "PB", "PC"],
            "Active Energy": ["EPA", "EPB", "EPC", "EPSum"],
            "Reactive Energy": ["EQA", "EQB", "EQC", "EQSum"],
            "Apparent Energy": ["ESA", "ESB", "ESC", "ESSum"],
        }

        # Calculate max and min values for each section
        for section, columns in sections.items():
            if all(col in df.columns for col in columns):
                section_df = df[columns]
                max_dfs.append(section_df.max(axis=0).rename(section))
                min_dfs.append(section_df.min(axis=0).rename(section))

        # Process Total Harmonic Distortion (THD) metrics
        thd_sections = {
            "UTHD": ["UTHA", "UTHB", "UTHC", "UTHAvg"],
            "ITHD": ["ITHA", "ITHB", "ITHC", "ITHAvg"],
            "ITHD3": ["ITHXA", "ITHXB", "ITHXC"],
            "ITHD5": ["ITHYA", "ITHYB", "ITHYC"],
            "ITHD7": ["ITHZA", "ITHZB", "ITHZC"],
        }

        for section, columns in thd_sections.items():
            if all(col in df.columns for col in columns):
                section_df = df[columns]
                max_dfs.append(section_df.max(axis=0).rename(section))
                min_dfs.append(section_df.min(axis=0).rename(section))

        # Combine all max/min values
        max_vals = pd.concat(max_dfs, axis=0) if max_dfs else pd.DataFrame()
        min_vals = pd.concat(min_dfs, axis=0) if min_dfs else pd.DataFrame()

        # Calculate energy consumption (difference between start and end)
        energy_sections = {
            "Active Energy": ["EPA", "EPB", "EPC", "EPSum"],
            "Reactive Energy": ["EQA", "EQB", "EQC", "EQSum"],
            "Apparent Energy": ["ESA", "ESB", "ESC", "ESSum"],
        }

        energy_changes = []
        for section, columns in energy_sections.items():
            if all(col in df.columns for col in columns):
                # Get first and last readings
                start = df[columns].head(1)
                end = df[columns].tail(1).reset_index(drop=True)
                
                # Calculate the change (consumption)
                section_change = pd.concat([start, end, end - start], axis=0).T
                section_change.columns = ["First", "Last", "Change"]
                section_change["Category"] = section
                energy_changes.append(section_change)

        # Combine all energy summaries
        energy_summary = pd.concat(energy_changes, axis=0) if energy_changes else pd.DataFrame()
        logger.info("Descriptive statistics calculated successfully")
        return max_vals, min_vals, energy_summary


def performance_monitor(func):
    """
    Decorator to measure execution time, memory usage, and CPU load of a function.
    
    This wrapper provides detailed performance metrics including:
    - Execution time
    - Memory usage change
    - Peak memory usage
    - CPU utilization
    - Function profiling stats (top 10 most time-consuming operations)
    
    Args:
        func: The function to monitor
        
    Returns:
        Wrapped function with performance monitoring
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Initialize monitoring tools
        process = psutil.Process()
        profiler = cProfile.Profile()
        profiler.enable()
        tracemalloc.start()
        
        # Capture starting metrics
        start_time = time.time()
        start_memory = process.memory_info().rss / 1024**2  # MB

        # Execute the function
        result = func(*args, **kwargs)

        # Capture ending metrics
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024**2  # MB
        current, peak = tracemalloc.get_traced_memory()
        peak_memory = peak / 1024**2  # MB
        tracemalloc.stop()

        # Print profiling results
        profiler.disable()
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumtime")
        ps.print_stats(10)  # Show top 10 functions by cumulative time
        print("Profiling stats:\n", s.getvalue())

        # Display performance summary
        print(f"Function: {func.__name__}")
        print(f"Execution Time: {end_time - start_time:.4f} sec")
        print(f"Memory Usage Change: {end_memory - start_memory:.4f} MB")
        print(f"Peak Memory Usage: {peak_memory:.4f} MB")
        print(f"CPU Usage: {process.cpu_percent(interval=0.1)}%\n")
        
        return result

    return wrapper

def main(path: str):
    """
    Main function to process and analyze power data.
    
    This function orchestrates the entire data processing pipeline:
    1. Loads data from the specified CSV file
    2. Preprocesses the data
    3. Creates visualizations
    4. Optionally calculates descriptive statistics
    
    Args:
        path: Path to the CSV data file
    """
    # Load data
    df = load_data(path)
    if df is None:
        logger.error("Failed to load data. Exiting.")
        return

    # Preprocess data
    df = preprocess_data(df)
    if df.empty:
        logger.error("Data preprocessing failed. Exiting.")
        return

    # Plot time series
    plot_time_series(
        df, ["IA", "IB", "IC"], "Current Time Series", 
        min_max_arrows=True, downsample=5
    )
    plot_time_series(
        df, ['PA', 'PB', 'PC'], 'Power Analysis', 
        min_max_arrows=True
    )

    # plot_gaussian_distribution(
    #     df,
    #     columns=["UA", "UB", "UC"],
    #     title="Voltage Distribution Analysis",
    #     bins=250,
    #     # start_date="2024-10-20, 14:00",
    #     # end_date="2024-10-23, 14:00",
    # )

    # Descriptive statistics
    # describe_data(df)


@performance_monitor  # load: 7-9 sec; convert to datetime: 10 sec
def load_data(path: str, rows: int = 0) -> Optional[pd.DataFrame]:
    data_loader = DataLoader()
    return data_loader.load_data(path, rows)


@performance_monitor
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    processor = DataProcessor()
    return processor.preprocess(df)


@performance_monitor
def describe_data(df: pd.DataFrame) -> None:
    stats = DescriptiveStats()
    max_vals, min_vals, p_max_sum = stats.describe(df)
    # print(pd.DataFrame({'max': max_vals, 'min': min_vals}))
    # print(f'Power max sum: {p_max_sum}')


@performance_monitor
def plot_time_series(
    df: pd.DataFrame,
    columns: List[str],
    title: str,
    min_max_arrows: bool = False,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    downsample: int = 1,
) -> None:
    plotter = Plotter()
    plotter.plot_time_series(
        df, columns, title, min_max_arrows, start_date, end_date, downsample
    )


@performance_monitor
def plot_gaussian_distribution(
    df: pd.DataFrame,
    columns: List[str],
    title: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    bins: int = 50,
) -> None:
    plotter = Plotter()
    plotter.plot_gaussian_distribution(df, columns, title, start_date, end_date, bins)


# endregion
if __name__ == "__main__":
    PATH = "data\DataSheet_1819011001_3P4W_3.csv"
    # PATH = "data\Copy of DataSheet_1819011001_3P4W-ХХХ.csv" # million rows
    
    main(path=PATH)
