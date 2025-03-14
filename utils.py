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
    def __init__(self):
        # Define dtypes for each column based on your header row.
        # Numeric columns are set to float32 to reduce memory usage.
        self.dtype = {
            "Date": "object",
            "Time": "object",
            "UA": "float32",
            "UB": "float32",
            "UC": "float32",
            "UAvg": "float32",
            "UTHA": "float32",
            "UTHB": "float32",
            "UTHC": "float32",
            "UTHAvg": "float32",
            "IA": "float32",
            "IB": "float32",
            "IC": "float32",
            "IAvg": "float32",
            "ITHA": "float32",
            "ITHB": "float32",
            "ITHC": "float32",
            "ITHAvg": "float32",
            "ITHXA": "float32",
            "ITHXB": "float32",
            "ITHXC": "float32",
            "ITHYA": "float32",
            "ITHYB": "float32",
            "ITHYC": "float32",
            "ITHZA": "float32",
            "ITHZB": "float32",
            "ITHZC": "float32",
            "FA": "float32",
            "FB": "float32",
            "FC": "float32",
            "FAvg": "float32",
            "PFA": "float32",
            "PFB": "float32",
            "PFC": "float32",
            "PFAvg": "float32",
            "PA": "float32",
            "PB": "float32",
            "PC": "float32",
            "PSum": "float32",
            "QA": "float32",
            "QB": "float32",
            "QC": "float32",
            "QSum": "float32",
            "SA": "float32",
            "SB": "float32",
            "SC": "float32",
            "SSum": "float32",
            "EPA": "float32",
            "EPB": "float32",
            "EPC": "float32",
            "EPSum": "float32",
            "EQA": "float32",
            "EQB": "float32",
            "EQC": "float32",
            "EQSum": "float32",
            "ESA": "float32",
            "ESB": "float32",
            "ESC": "float32",
            "ESSum": "float32",
            "DmIA": "float32",
            "DmIB": "float32",
            "DmIC": "float32",
            "DmIAVG": "float32",
            "PDmIA": "float32",
            "PDmIA_D/T": "object",
            "PDmIB": "float32",
            "PDmIB_D/T": "object",
            "PDmIC": "float32",
            "PDmIC_D/T": "object",
            "PDmIAVG": "float32",
            "PDmIAVG_D/T": "object",
            "DmP": "float32",
            "PDmP": "float32",
            "PDmP_D/T": "object",
            "DmQ": "float32",
            "PDmQ": "float32",
            "PDmQ_D/T": "object",
            "DmS": "float32",
            "PDmS": "float32",
            "PDmS_D/T": "object",
        }

    def load_data(
        self, file_path: str, rows: int = 0, usecols: list[str] = None
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
                # parse_dates=[[0, 1]], (time bottleneck, better not use)
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

    # (Time bottleneck, better not user)
    def combine_date_time(
        self,
        df: pd.DataFrame,
        date_col: str = "Date",
        time_col: str = "Time",
        new_col: str = "DateTime",
    ) -> pd.DataFrame:
        """Combines separate date and time columns into one datetime column.

        Args:
            df (pd.DataFrame): DataFrame containing separate date and time columns.
            date_col (str): Name of the date column.
            time_col (str): Name of the time column.
            new_col (str): Name for the combined datetime column.

        Returns:
            pd.DataFrame: DataFrame with the new datetime column.
        """
        try:
            df[new_col] = pd.to_datetime(
                df[date_col] + " " + df[time_col], errors="coerce"
            )
            return df
        except Exception as e:
            logger.error(f"Failed to combine date and time columns: {e}")
            return df


class DataProcessor:
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocesses the data by:
        - taking the absolute values of the power columns.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.

        Returns:
            df (pd.DataFrame): The preprocessed DataFrame.
        """
        if df is None or df.empty:
            logger.error("No DataFrame provided to preprocess.")
            return pd.DataFrame()
        try:
            df[["PA", "PB", "PC"]] = df[
                ["PA", "PB", "PC"]
            ].abs()  # taking aboslute values to remove negative values
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
    ):
        """Plots a time series chart for the given columns and returns the figure."""
        if df is None or df.empty:
            logger.error("Empty DataFrame provided to plot on time series chart.")
            return None

        if start_date or end_date:
            df = self._filter_date_range(df, start_date, end_date)
        df = self._fill_missing_time_with_zeros(df, columns)

        self.fig = go.Figure()
        for idx, col in enumerate(columns):
            self.fig.add_trace(
                go.Scatter(
                    x=df["Date_Time"],
                    y=df[col],
                    mode="lines",
                    name=col,
                    line=dict(color=LINE_COLORS[idx]),
                )
            )

        self.fig.update_layout(
            title=title,
            xaxis_title="Date and Time",
            yaxis_title="Values",
            hovermode="closest",
            title_font=dict(
                size=16, color="black"
            ),  # Increased font size for readability
            xaxis=dict(title_font=dict(size=14), tickfont=dict(size=12)),
            yaxis=dict(title_font=dict(size=14), tickfont=dict(size=12)),
            legend=dict(font=dict(size=12)),
        )

        if min_max_arrows:
            self._add_min_max_arrows(df, columns)

        self.fig.show()
        return self.fig  # Return figure instead of showing it

    # def plot_gaussian_distribution(self, df: pd.DataFrame, columns: list[str], title: str, start_date: str = None, end_date: str = None):
    #     '''Plots a Gaussian distribution chart for the given columns using matplotlib, excluding zero values.'''
    #     if df is None or df.empty:
    #         logger.error("Empty DataFrame provided for Gaussian distribution plot.")
    #         return None

    #     # Filter DataFrame based on date range if specified
    #     df = self._filter_date_range(df, start_date, end_date)

    #     # Set up the matplotlib figure
    #     plt.figure(figsize=(10, 6))

    #     for col in columns:
    #         # Exclude zero values from the data
    #         non_zero_data = df[col][df[col] != 0]

    #         if non_zero_data.empty:
    #             logger.warning(f"All values are zero in column {col}, skipping plot.")
    #             continue

    #         sns.histplot(non_zero_data, kde=True, bins=30, label=col, stat="density", alpha=0.6)

    #     # Configure the plot title and labels
    #     plt.title(title, fontsize=16, color='black')
    #     plt.xlabel("Value", fontsize=14)
    #     plt.ylabel("Density", fontsize=14)
    #     plt.xticks(fontsize=12)
    #     plt.yticks(fontsize=12)
    #     plt.legend(title="Columns", fontsize=12, title_fontsize=12)
    #     plt.grid(True, linestyle="--", alpha=0.7)

    #     # Show the plot
    #     plt.tight_layout()
    #     plt.show()

    def plot_gaussian_distribution(
        self,
        df: pd.DataFrame,
        columns: list[str],
        title: str,
        start_date: str = None,
        end_date: str = None,
    ):
        """Plots a Gaussian distribution chart for the given columns and returns the figure."""
        if df is None or df.empty:
            logger.error("Empty DataFrame provided for Gaussian distribution plot.")
            return None

        # hist_data = [df[col] for col in columns]
        hist_data = [df[col][df[col] != 0] for col in columns]
        group_labels = columns

        df = self._filter_date_range(df, start_date, end_date)

        self.fig = ff.create_distplot(hist_data, group_labels, bin_size=0.2)

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

    def _filter_date_range(
        self, df: pd.DataFrame, start_date: str = None, end_date: str = None
    ) -> pd.DataFrame:
        if df is None or df.empty:
            logger.error("Empty DataFrame provided to filter date range.")
            return pd.DataFrame()

        if start_date or end_date:
            try:
                df_copy = df.copy()

                if start_date:
                    start_date = pd.to_datetime(start_date)
                    df_copy = df_copy[(df["Date_Time"] >= start_date)]

                if end_date:
                    end_date = pd.to_datetime(end_date)
                    df_copy = df_copy[(df["Date_Time"] <= end_date)]

                return df_copy
            except ValueError as e:
                logger.error(f"Failed to filter date range: {e}")
                return df

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

    # Convert to datetime
    df = convert_to_datetime(df)

    # Preprocess data
    # df = preprocess_data(df)

    # Plot time series
    # plot_time_series(df, ["IA", "IB", "IC"], "Current", min_max_arrows=True)
    # plot_time_series(df, ['VA', 'VB', 'VC'], 'Voltage', min_max_arrows=True)
    # plot_time_series(df, ['PA', 'PB', 'PC'], 'Power', min_max_arrows=True)

    # Plot gaussian distribution
    # plot_gaussian_distribution(df, ["IA", "IB", "IC"], "Current")

    # Descriptive statistics
    # describe_data(df)


@performance_monitor  # 7-9 sec
def load_data(path: str, rows: int = 0) -> pd.DataFrame | None:
    data_loader = DataLoader()
    return data_loader.load_data(path, rows)


@performance_monitor  # 10 sec
def convert_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    data_loader = DataLoader()
    return data_loader.combine_date_time(df)


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
    columns: list[str],
    title: str,
    min_max_arrows: bool = False,
    start_date: str = None,
    end_date: str = None,
) -> None:
    plotter = Plotter()
    plotter.plot_time_series(df, columns, title, min_max_arrows, start_date, end_date)

    # plotter.plot_time_series(df, columns=['IA', 'IB', 'IC'], title="Current Timeline", min_max_arrows=True, start_date='2024-10-20')
    # plotter.plot_time_series(df, columns=['PA', 'PB', 'PC'], title="Power Timeline", min_max_arrows=True, start_date='2024-10-20')
    plotter.plot_gaussian_distribution(
        df,
        columns=["IA", "IB", "IC"],
        title="Gaussian Distribution",
        start_date="2024-10-20, 14:00",
        end_date="2024-10-23, 14:00",
    )


@performance_monitor
def plot_gaussian_distribution(
    df: pd.DataFrame,
    columns: list[str],
    title: str,
    start_date: str = None,
    end_date: str = None,
) -> None:
    plotter = Plotter()
    plotter.plot_gaussian_distribution(df, columns, title, start_date, end_date)


# endregion
if __name__ == "__main__":
    # PATH = 'data\DataSheet_1819011001_3P4W_3-1.csv'
    PATH = "data\Copy of DataSheet_1819011001_3P4W-ХХХ.csv"
    main(path=PATH)
