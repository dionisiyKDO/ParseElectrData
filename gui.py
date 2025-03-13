import pandas as pd
import numpy as np
import scipy

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import sv_ttk
from ttkthemes import ThemedTk

from os import path
from utils import *


class FileSelectorGUI:
    def __init__(self, master):
        self.master = master
        self.master.geometry("450x150")
        self.master.title("File Selector")

        self.file_path_label = ttk.Label(self.master, text="Please, select the file...")
        self.file_path_label.pack(pady=10)

        self.select_file_button = ttk.Button(self.master, text="Select File", command=self.select_file)
        self.select_file_button.pack(pady=2, padx=10, fill="x")

        self.process_file_button = ttk.Button(self.master, text="Process File", command=self.process_file)
        self.process_file_button.pack(pady=2, padx=10, fill="x")

        self.selected_file_path = ""
        self.child_window = None

    def select_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.selected_file_path = file_path
            self.update_file_info()

    def update_file_info(self):
        if self.selected_file_path:
            file_name = path.basename(self.selected_file_path)
            label_text = f"Selected file -  {file_name}"
            self.file_path_label.config(text=label_text)

    def process_file(self):
        if self.selected_file_path and self.child_window is None:
            self.child_window = tk.Toplevel(self.master)
            self.child_window.protocol('WM_DELETE_WINDOW', self.close_Toplevel)
            child_app = ChildWindow(self.child_window, self.selected_file_path)
    
    def close_Toplevel(self):
        self.child_window.destroy()
        self.child_window = None

class ChildWindow:
    def __init__(self, master, path):
        self.master = master
        self.master.geometry("1055x350")  # Increased width to accommodate two Text widgets
        self.master.title("Make plots")

        self.data_loader = DataLoader()
        self.processor = DataProcessor()
        self.plotter = Plotter()
        self.stats = DescriptiveStats()

        self.df = self.processor.preprocess(self.data_loader.load_data(path))

        # Create a PanedWindow
        paned_window = tk.PanedWindow(master, orient=tk.HORIZONTAL)
        paned_window.pack(expand=True, fill="both")

        # Left Frame for buttons
        left_frame = ttk.Frame(paned_window)
        paned_window.add(left_frame)

        window_label = ttk.Label(left_frame, text="Plots", font=("Helvetica", 14, "bold"))
        window_label.pack(pady=15)

        label_y_padding = (5, 0)
        button_x_padding = 6
        button_y_padding = 5

        # send df(dt, ia, ib, ic)
        curtm_label = ttk.Label(left_frame, text="Current timeline (IA IB IC)")
        curtm_label.pack(pady=label_y_padding)
        curtm_plot_button = ttk.Button(left_frame, text="Make plot",
                                       command=lambda: self.plotter.plot_time_series(self.df[['Date_Time', 'IA', 'IB', 'IC']],
                                                                                     columns=['IA', 'IB', 'IC'],
                                                                                     title="Current Timeline",
                                                                                     min_max_arrows=self.min_max_var.get()))
        curtm_plot_button.pack(fill='x', padx=button_x_padding, pady=button_y_padding)

        # send df(dt, pa, pb, pc)
        powtm_label = ttk.Label(left_frame, text="Power timeline (SA SB SC)")
        powtm_label.pack(pady=label_y_padding)
        powtm_plot_button = ttk.Button(left_frame, text="Make plot",
                                       command=lambda: self.plotter.plot_time_series(self.df[['Date_Time', 'SA', 'SB', 'SC']],
                                                                                     columns=['SA', 'SB', 'SC'],
                                                                                     title="Power Timeline",
                                                                                     min_max_arrows=self.min_max_var.get()))
        powtm_plot_button.pack(fill='x', padx=button_x_padding, pady=button_y_padding)

        # send df(dt, epa, epb, epc, epsum)
        energy_label = ttk.Label(left_frame, text="Active energy timeline (EPA EPB EPC)")
        energy_label.pack(pady=label_y_padding)
        energy_plot_button = ttk.Button(left_frame, text="Make plot",
                                        command=lambda: self.plotter.plot_time_series(self.df[['Date_Time', 'EPA', 'EPB', 'EPC', 'EPSum']],
                                                                                      columns=['EPA', 'EPB', 'EPC', 'EPSum'],
                                                                                      title="Active Energy Timeline",
                                                                                      min_max_arrows=self.min_max_var.get()))
        energy_plot_button.pack(fill='x', padx=button_x_padding, pady=button_y_padding)

        # min max arrows
        self.min_max_var = tk.BooleanVar()
        min_max_checkbox = ttk.Checkbutton(left_frame, text="Show Min Max Arrows", variable=self.min_max_var)
        min_max_checkbox.pack()

        separator = ttk.Separator(left_frame, orient='horizontal')
        separator.pack(fill='x', pady=8)

        # send df(ua, ub, uc)
        voltage_label = ttk.Label(left_frame, text="Voltage GaussDistr (UA UB UC)")
        voltage_label.pack(pady=label_y_padding)
        voltage_plot_button = ttk.Button(left_frame, text="Make plot",
                                          command=lambda: self.plotter.plot_gaussian_distribution(self.df[['UA', 'UB', 'UC']],
                                                                                                 columns=['UA', 'UB', 'UC'],
                                                                                                 title='Voltage Gaussian Distribution'))
        voltage_plot_button.pack(fill='x', padx=button_x_padding, pady=button_y_padding)

        # Right Frame for Statistics Display
        right_frame = ttk.Frame(paned_window)
        paned_window.add(right_frame)

        # Create Two Text Widgets
        text_box_min_max = tk.Text(right_frame, wrap=tk.WORD, height=20, width=46)
        text_box_min_max.grid(row=0, column=0, padx=5, pady=5)

        text_box_energy = tk.Text(right_frame, wrap=tk.WORD, height=20, width=51)
        text_box_energy.grid(row=0, column=1, padx=5, pady=5)

        # Populate the Min/Max Text Box
        max_vals, min_vals, energy_summary = self.stats.describe(self.df)

        if not max_vals.empty and not min_vals.empty:
            min_max_text = "Stats (Min/Max):\n"
            min_max_text += "-" * 45 + "\n"  # Increased separator width

            stats_df = pd.concat([max_vals, min_vals], axis=1)
            stats_df.columns = ["Max", "Min"]

            # Adjusted column widths for better alignment
            param_width = 15   # Parameter column
            max_width = 15     # Max values column
            min_width = 15     # Min values column
            
            # Header row with better alignment
            min_max_text += f"{'Parameter'.ljust(param_width)}{'Max'.rjust(max_width)}{'Min'.rjust(min_width)}\n"
            min_max_text += "-" * 45 + "\n"  # Increased separator width again

            # Format each row with consistent spacing
            for index, row in stats_df.iterrows():
                # Using better number formatting with fewer decimals for large numbers
                max_val = f"{row['Max']:.2f}" if row['Max'] < 1000 else f"{row['Max']:.0f}"
                min_val = f"{row['Min']:.2f}" if row['Min'] < 1000 else f"{row['Min']:.0f}"
                
                # Ensure consistent column widths with proper padding
                min_max_text += f"{index.ljust(param_width)}{max_val.rjust(max_width)}{min_val.rjust(min_width)}\n"

            text_box_min_max.insert(tk.END, min_max_text)

        # Populate the Energy Text Box
        if not energy_summary.empty:
            energy_text = "Energy Changes:\n"
            energy_text += "-" * 50 + "\n"

            column_width = 10
            energy_text += f"{'Category'.ljust(20)}{'First'.rjust(column_width)}{'Last'.rjust(column_width)}{'Change'.rjust(column_width)}\n"
            energy_text += "-" * 50 + "\n"

            for _, row in energy_summary.iterrows():
                energy_text += (f"{row['Category'].ljust(20)}"
                                f"{row['First']:>10.0f}"
                                f"{row['Last']:>10.0f}"
                                f"{row['Change']:>10.0f}\n")

            text_box_energy.insert(tk.END, energy_text)



if __name__ == "__main__":
    # root = ThemedTk(theme="arc")
    root = tk.Tk()
    # root.iconbitmap('folder_icon.ico')
    sv_ttk.set_theme("dark")  # Set theme for sv_ttk
    app = FileSelectorGUI(root)
    
    root.mainloop()
