from os.path import exists
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.figure_factory as ff

from typing import Tuple, Optional, Dict, List, Union, Iterator, Callable
from functools import lru_cache, wraps
from datetime import datetime
import gc

import time
import tracemalloc
import psutil
import cProfile
import pstats
import io

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Constants for colors
LINE_COLORS = ["yellow", "green", "red", "blue"]
ARROW_COLORS = ["black", "black", "black", "black"]
DEFAULT_CHUNK_SIZE = 100000  # Process 100k rows at a time


class DataLoader:
    @staticmethod
    def load_data(
        file_path: str, rows: int = 0, usecols: list[str] = None
    ) -> pd.DataFrame | None:
        """Loads data from a CSV file and returns a DataFrame.

        Args:
            file_path (str): The path to the CSV file.
            rows (int, optional): The number of rows to read from the file. Defaults to 0.
            usecols (list[str], optional): List of columns to load.
                If not None, only these columns (plus Date and Time if not already included) will be read.

        Returns:
            df (pd.DataFrame | None): The DataFrame containing the data. None if the file could not be loaded.
        """
        if not exists(file_path):
            logger.error("Incorrect file path provided.")
            return None

        try:
            # If usecols is specified, ensure Date and Time are included for later plotting
            if usecols is not None:
                usecols = list(set(usecols) | {"Date", "Time"})

            df = pd.read_csv(
                file_path,
                skiprows=2,
                nrows=rows if rows else None,
                low_memory=True,
                usecols=usecols,
                parse_dates=[[0, 1]], # (time bottleneck, better not use)
            )

            if df is None or df.empty:
                logger.error("Empty DataFrame loaded in load_data().")
                return None

            # Drop any unwanted unnamed columns that may have been created.
            df.drop(
                columns=[col for col in df.columns if col.startswith("Unnamed")],
                inplace=True,
                errors="ignore",
            )

            logger.info("Data loaded successfully.")
            return df
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return None


class DataProcessor:
    @staticmethod
    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the data by:
        - taking the absolute values of the power columns.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            df (pd.DataFrame): Preprocessed DataFrame
        """
        if df is None or df.empty:
            logger.error("Empty DataFrame provided for preprocessing.")
            return pd.DataFrame()

        try:
            power_cols = ["PA", "PB", "PC"]
            df[power_cols] = df[power_cols].abs()

            logger.info("Data preprocessing completed.")
            return df

        except KeyError as e:
            logger.error(f"Preprocessing failed due to missing columns: {e}")
            return pd.DataFrame()


class Plotter:
    def __init__(self):
        self.fig = None

    def plot_time_series(
        self,
        df: pd.DataFrame,
        columns: list[str],
        title: str,
        min_max_arrows: bool = False,
        start_date: str = None,
        end_date: str = None,
        downsample: int = 1,  # Optional downsampling factor (1 = no downsampling)
    ) -> go.Figure:
        """
        Plots a time series chart for the given columns.

        Args:
            df (pd.DataFrame): DataFrame containing time series data
            columns (list[str]): Column names to plot
            title (str): Chart title
            min_max_arrows (bool): Whether to add min/max annotations
            start_date (str): Optional start date filter
            end_date (str): Optional end date filter

        Returns:
            go.Figure: Plotly figure object
        """
        if df.empty:
            logger.error("Empty DataFrame provided for time series plot.")
            return None

        if start_date or end_date:
            df = self._filter_date_range(df, start_date, end_date)

        df = self._fill_missing_time_with_zeros(df, columns)

        # Optional downsampling to reduce rendering load if needed
        if downsample > 1:
            df = df.iloc[::downsample].copy()

        self.fig = go.Figure()

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

            # self.fig.add_trace(
            #     go.Scattergl(
            #         x=df["Date_Time"],
            #         y=df[col],
            #         mode="lines",
            #         name=col,
            #         line=dict(color=LINE_COLORS[idx % len(LINE_COLORS)]),
            #     )
            # )

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

    # def plot_gaussian_distribution(
    #     self,
    #     df: pd.DataFrame,
    #     columns: list[str],
    #     title: str,
    #     start_date: str = None,
    #     end_date: str = None,
    #     bins: int = 50,
    # ):
    #     """Plots a Gaussian distribution chart for the given columns and returns the figure."""
    #     if df is None or df.empty:
    #         logger.error("Empty DataFrame provided for Gaussian distribution plot.")
    #         return None

    #     # hist_data = [df[col] for col in columns]
    #     hist_data = [df[col][df[col] != 0] for col in columns]
    #     group_labels = columns

    #     df = self._filter_date_range(df, start_date, end_date)

    #     self.fig = ff.create_distplot(hist_data, group_labels, bin_size=0.2)

    #     self.fig.update_layout(
    #         title=title,
    #         xaxis_title="Value",
    #         yaxis_title="Density",
    #         title_font=dict(size=16, color="black"),
    #         xaxis=dict(title_font=dict(size=14), tickfont=dict(size=12)),
    #         yaxis=dict(title_font=dict(size=14), tickfont=dict(size=12)),
    #         legend=dict(title_text="Columns", font=dict(size=12)),
    #     )

    #     self.fig.show()
    #     return self.fig  # Return figure instead of showing it
    
    def plot_gaussian_distribution(
        self,
        df: pd.DataFrame,
        columns: list[str],
        title: str,
        start_date: str = None,
        end_date: str = None,
        bins: int = 50  # Number of bins for the histogram
    ):
        """Plots a Gaussian distribution chart using a pre-computed histogram.
        
        This avoids heavy kernel density estimation with large datasets.
        """
        if df is None or df.empty:
            logger.error("Empty DataFrame provided for Gaussian distribution plot.")
            return None

        # Filter the DataFrame first
        df = self._filter_date_range(df, start_date, end_date)
        self.fig = go.Figure()
        
        for idx, col in enumerate(columns):
            # Exclude zeros for the distribution
            data = df[col][df[col] != 0].dropna()
            if data.empty:
                logger.warning(f"No valid data found for column: {col}")
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
        self, df: pd.DataFrame, start_date: str = None, end_date: str = None
    ) -> pd.DataFrame:
        """Filters the DataFrame between start_date and end_date based on Date_Time."""
        if df is None or df.empty:
            logger.error("Empty DataFrame provided to filter date range.")
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
            logger.error(f"Failed to filter date range: {e}")
            return df
        return df_copy

    def _fill_missing_time_with_zeros(
        self, df: pd.DataFrame, columns: list
    ) -> pd.DataFrame:
        """Fills missing datetime values with zeros, especially for gaps greater than 2 hours."""
        if "Date_Time" not in df.columns:
            logger.error(
                "DataFrame does not contain 'Date_Time' column for filling missing times."
            )
            return df

        try:
            df["Date_Time"] = pd.to_datetime(df["Date_Time"])

            # Sort by Date_Time to ensure proper gap calculation
            df = df.sort_values(by="Date_Time").reset_index(drop=True)

            # Identify gaps
            df["Time_Diff"] = df["Date_Time"].diff()

            # Create a DataFrame for the new zero-filled rows
            zero_fill_rows = []

            for i in range(1, len(df)):
                if df["Time_Diff"].iloc[i] and df["Time_Diff"].iloc[i] > pd.Timedelta(
                    hours=2
                ):
                    start_time = df["Date_Time"].iloc[i - 1]
                    end_time = df["Date_Time"].iloc[i]

                    # Create new timestamps for 2-hour intervals within the gap
                    current_time = start_time + pd.Timedelta(hours=2)

                    while current_time < end_time:
                        zero_fill_row = {
                            col: 0 for col in columns
                        }  # Fill zeros for specified columns
                        zero_fill_row["Date_Time"] = current_time
                        zero_fill_rows.append(zero_fill_row)
                        current_time += pd.Timedelta(
                            hours=2
                        )  # Move to the next 2-hour interval

            # Convert zero_fill_rows to a DataFrame
            if zero_fill_rows:
                zero_fill_df = pd.DataFrame(zero_fill_rows)
                df = pd.concat([df, zero_fill_df], ignore_index=True)

            # Remove duplicates, fill missing columns with zeros, and sort again
            df = df.drop_duplicates(subset="Date_Time")
            df = df.sort_values(by="Date_Time").reset_index(drop=True)

            # Ensure that missing columns are added with zeros
            # for col in columns:
            #     if col not in df.columns:
            #         df[col] = 0
            #     df[col].fillna(0, inplace=True)

            # Drop the Time_Diff column as it's no longer needed
            df.drop(columns=["Time_Diff"], inplace=True)

            return df

        except Exception as e:
            logger.error(f"Failed to fill missing time with zeros: {e}")
            return df

    def _add_min_max_arrows(self, df: pd.DataFrame, columns: list):
        for idx, col in enumerate(columns):
            try:
                # Create a mask to filter out rows where the specified column value is less than or equal to 0
                mask = df[col] > 0.01

                # Use this mask to find the max and min rows while keeping Date_Time intact
                if mask.any():  # Check if there are any valid rows to work with
                    max_row = df[mask].loc[df[mask][col].idxmax()]
                    min_row = df[mask].loc[df[mask][col].idxmin()]
                    max_value = max_row[col]
                    min_value = min_row[col]

                    self._add_annotation(max_row, col, idx, max_value, "Max")
                    self._add_annotation(min_row, col, idx, min_value, "Min")
                else:
                    logger.warning(f"No valid data found for {col} greater than 0.01")
            except ValueError as e:
                logger.warning(f"Error finding max/min for {col}: {e}")

    def _add_annotation(
        self, row: pd.Series, col: str, idx: int, value: float, label: str
    ):
        """Adds a formatted annotation to the plot."""
        self.fig.add_annotation(
            text=f"<b>{label}:</b><br>{col} = {value:.2f}",  # Improved formatting for clarity
            x=row["Date_Time"],
            y=row[col],
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor=ARROW_COLORS[idx],
            ax=0,  # Center arrow horizontally
            ay=-40,  # Maintain vertical offset
            font=dict(
                size=12, color="black"
            ),  # Make font color more neutral for readability
            bgcolor="rgba(255, 255, 255, 0.66)",  # Semi-transparent white background for better contrast
            bordercolor="black",  # Border color for annotation box
            borderwidth=1,  # Border width for annotation box
            borderpad=3,  # Padding inside the annotation box
        )


class DescriptiveStats:
    @staticmethod
    def describe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if df.empty:
            logger.error("Empty DataFrame provided for descriptive statistics.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        max_dfs = []
        min_dfs = []

        # region Basic Metrics
        sections = {
            "Current": ["IA", "IB", "IC"],
            "Voltage": ["UA", "UB", "UC"],
            "Power": ["PA", "PB", "PC"],
            "Active Energy": ["EPA", "EPB", "EPC", "EPSum"],
            "Reactive Energy": ["EQA", "EQB", "EQC", "EQSum"],
            "Apparent Energy": ["ESA", "ESB", "ESC", "ESSum"],
        }

        for section, columns in sections.items():
            if all(col in df.columns for col in columns):
                section_df = df[columns]
                max_dfs.append(section_df.max(axis=0).rename(section))
                min_dfs.append(section_df.min(axis=0).rename(section))
        # endregion

        # region Harmonic Distortion (THD)
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
        # endregion

        # Combine min/max values
        max_vals = pd.concat(max_dfs, axis=0)
        min_vals = pd.concat(min_dfs, axis=0)

        # region Energy Changes (Start-End)
        energy_sections = {
            "Active Energy": ["EPA", "EPB", "EPC", "EPSum"],
            "Reactive Energy": ["EQA", "EQB", "EQC", "EQSum"],
            "Apparent Energy": ["ESA", "ESB", "ESC", "ESSum"],
        }

        energy_changes = []
        for section, columns in energy_sections.items():
            if all(col in df.columns for col in columns):
                start, end = (
                    df[columns].head(1),
                    df[columns].tail(1).reset_index(drop=True),
                )
                section_change = pd.concat([start, end, end - start], axis=0).T
                section_change.columns = ["First", "Last", "Change"]
                section_change["Category"] = section
                energy_changes.append(section_change)

        energy_summary = (
            pd.concat(energy_changes, axis=0) if energy_changes else pd.DataFrame()
        )
        # endregion

        logger.info("Descriptive statistics calculated.")
        return max_vals, min_vals, energy_summary

    def __min_max(self, df: pd.DataFrame, columns: list[str]) -> Tuple[dict, dict]:
        if df.empty:
            logger.error("Empty DataFrame provided for min/max values.")
            return {}, {}
        max_dfs = []
        min_dfs = []

        # region Basic
        # Current
        df_I = df.loc[:, ["IA", "IB", "IC"]]
        df_I_min = df_I.min(axis=0).rename("max", inplace=True)
        df_I_max = df_I.max(axis=0).rename("min", inplace=True)
        min_dfs.append(df_I_min)
        max_dfs.append(df_I_max)

        # Voltage
        df_U = df.loc[:, ["UA", "UB", "UC"]]
        df_U_min = df_U.min(axis=0).rename("max", inplace=True)
        df_U_max = df_U.max(axis=0).rename("min", inplace=True)
        min_dfs.append(df_U_min)
        max_dfs.append(df_U_max)

        # Power
        df_P = df.loc[:, ["PA", "PB", "PC"]]
        df_P_min = df_P.min(axis=0).rename("max", inplace=True)
        df_P_max = df_P.max(axis=0).rename("min", inplace=True)
        min_dfs.append(df_P_min)
        max_dfs.append(df_P_max)
        # endregion

        # region Energy
        # Active energy
        df_EP_cols = ["EPA", "EPB", "EPC", "EPSum"]
        df_EP = df.loc[:, ["EPA", "EPB", "EPC", "EPSum"]]
        df_EP_min = df_EP.min(axis=0).rename("max", inplace=True)
        df_EP_max = df_EP.max(axis=0).rename("min", inplace=True)
        min_dfs.append(df_EP_min)
        max_dfs.append(df_EP_max)

        # Reactive energy
        df_EQ_cols = ["EQA", "EQB", "EQC", "EQSum"]
        df_EQ = df.loc[:, ["EQA", "EQB", "EQC", "EQSum"]]
        df_EQ_min = df_EQ.min(axis=0).rename("max", inplace=True)
        df_EQ_max = df_EQ.max(axis=0).rename("min", inplace=True)
        min_dfs.append(df_EQ_min)
        max_dfs.append(df_EQ_max)

        # Apparent energy
        df_ES_cols = ["ESA", "ESB", "ESC", "ESSum"]
        df_ES = df.loc[:, ["ESA", "ESB", "ESC", "ESSum"]]
        df_ES_min = df_ES.min(axis=0).rename("max", inplace=True)
        df_ES_max = df_ES.max(axis=0).rename("min", inplace=True)
        min_dfs.append(df_ES_min)
        max_dfs.append(df_ES_max)
        # endregion

        # region Power
        # Active Power(W)
        df_PA_cols = ["PA", "PB", "PC", "PSum"]
        df_PA = df.loc[:, ["PA", "PB", "PC", "PSum"]]
        df_PA_min = df_PA.min(axis=0).rename("max", inplace=True)
        df_PA_max = df_PA.max(axis=0).rename("min", inplace=True)
        min_dfs.append(df_PA_min)
        max_dfs.append(df_PA_max)

        # Reactive Power(Var)
        df_PB_cols = ["QA", "QB", "QC", "QSum"]
        df_PB = df.loc[:, ["QA", "QB", "QC", "QSum"]]
        df_PB_min = df_PB.min(axis=0).rename("max", inplace=True)
        df_PB_max = df_PB.max(axis=0).rename("min", inplace=True)
        min_dfs.append(df_PB_min)
        max_dfs.append(df_PB_max)

        # Apparent Power(Va)
        df_PC_cols = ["SA", "SB", "SC", "SSum"]
        df_PC = df.loc[:, ["SA", "SB", "SC", "SSum"]]
        df_PC_min = df_PC.min(axis=0).rename("max", inplace=True)
        df_PC_max = df_PC.max(axis=0).rename("min", inplace=True)
        min_dfs.append(df_PC_min)
        max_dfs.append(df_PC_max)
        # endregion

        df_max = pd.concat(min_dfs, axis=0)
        df_min = pd.concat(max_dfs, axis=0)

        result = pd.concat([df_max, df_min], axis=1)
        result.columns = ["max", "min"]

        print(f"{result}")


def performance_monitor(func):
    """
    Decorator to measure execution time, memory usage, and CPU load of a function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        profiler = cProfile.Profile()
        profiler.enable()
        tracemalloc.start()
        start_time = time.time()
        start_memory = process.memory_info().rss / 1024**2  # MB

        result = func(*args, **kwargs)

        end_time = time.time()
        end_memory = process.memory_info().rss / 1024**2  # MB
        current, peak = tracemalloc.get_traced_memory()
        peak_memory = peak / 1024**2  # MB
        tracemalloc.stop()

        profiler.disable()
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumtime")
        ps.print_stats(10)  # Top 10 functions
        print("Profiling stats:\n", s.getvalue())

        print(f"Function: {func.__name__}")
        print(f"Execution Time: {end_time - start_time:.4f} sec")
        print(f"Memory Usage Change: {end_memory - start_memory:.4f} MB")
        print(f"Peak Memory Usage: {peak_memory:.4f} MB")
        print(f"CPU Usage: {process.cpu_percent(interval=0.1)}%\n")
        return result

    return wrapper

    return wrapper


# region Main
# @performance_monitor
def main(path: str):
    # Load data
    df = load_data(path)

    # Preprocess data
    # df = preprocess_data(df)

    # Plot time series
    # plot_time_series(df, ["IA", "IB", "IC"], "Current", min_max_arrows=True, downsample=5)
    # plot_time_series(df, ['PA', 'PB', 'PC'], 'Power', min_max_arrows=True)

    # Plot gaussian distribution
    # plot_gaussian_distribution(
    #     df,
    #     columns=["UA", "UB", "UC"],
    #     title="Gaussian Distribution",
    #     bins=250,
    #     # start_date="2024-10-20, 14:00",
    #     # end_date="2024-10-23, 14:00",
    # )

    # Descriptive statistics
    # describe_data(df)


@performance_monitor  # load: 7-9 sec; convert to datetime: 10 sec
def load_data(path: str, rows: int = 0) -> pd.DataFrame | None:
    data_loader = DataLoader()
    return data_loader.load_data(path, rows)

# @performance_monitor
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
    columns: list[str],
    title: str,
    min_max_arrows: bool = False,
    start_date: str = None,
    end_date: str = None,
    downsample: int = 1,
) -> None:
    plotter = Plotter()
    plotter.plot_time_series(df, columns, title, min_max_arrows, start_date, end_date, downsample)

@performance_monitor
def plot_gaussian_distribution(
    df: pd.DataFrame,
    columns: list[str],
    title: str,
    start_date: str = None,
    end_date: str = None,
    bins: int = 50,
) -> None:
    plotter = Plotter()
    plotter.plot_gaussian_distribution(df, columns, title, start_date, end_date, bins)


# endregion
if __name__ == "__main__":
    # PATH = 'data\DataSheet_1819011001_3P4W_3-1.csv'
    PATH = "data\Copy of DataSheet_1819011001_3P4W-ХХХ.csv"
    main(path=PATH)
