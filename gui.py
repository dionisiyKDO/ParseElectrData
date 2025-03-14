import pandas as pd
import tkinter as tk
import sv_ttk
from tkinter import ttk
from tkinter import filedialog
from os import path

from utils import *


class FileSelectorGUI:
    def __init__(self, master):
        self.master = master
        self.master.geometry("450x250")
        self.master.title("File Selector")
        self.master.resizable(False, False)

        # Main frame to center elements
        main_frame = ttk.Frame(self.master, padding=15)
        main_frame.pack(expand=True)

        # File selection label
        self.file_path_label = ttk.Label(main_frame, text="Please, select the file...", anchor="center")
        self.file_path_label.grid(row=0, column=0, columnspan=2, pady=(5, 10))

        # File selection button
        self.select_file_button = ttk.Button(
            main_frame, text="Select File", command=self.select_file
        )
        self.select_file_button.grid(row=1, column=0, columnspan=2, pady=5, sticky="ew")

        # Start datetime label & entry
        self.start_label = ttk.Label(main_frame, text="Start Datetime:")
        self.start_label.grid(row=2, column=0, sticky="e", pady=5, padx=(10, 5))

        self.start_entry = ttk.Entry(main_frame, width=25, justify="center")
        self.start_entry.insert(0, "2024-10-23, 10:22:00")
        self.start_entry.grid(row=2, column=1, pady=5, padx=(5, 10))

        # End datetime label & entry
        self.end_label = ttk.Label(main_frame, text="End Datetime:")
        self.end_label.grid(row=3, column=0, sticky="e", pady=5, padx=(10, 5))

        self.end_entry = ttk.Entry(main_frame, width=25, justify="center")
        self.end_entry.insert(0, "2024-10-23, 12:00:00")
        self.end_entry.grid(row=3, column=1, pady=5, padx=(5, 10))

        # Process file button
        self.process_file_button = ttk.Button(
            main_frame, text="Process File", command=self.process_file
        )
        self.process_file_button.grid(row=4, column=0, columnspan=2, pady=15, sticky="ew")

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
            self.file_path_label.config(text=f"Selected file - {file_name}")

    def process_file(self):
        if self.selected_file_path and self.child_window is None:
            start_datetime = self.start_entry.get().strip()
            end_datetime = self.end_entry.get().strip()
            
            if start_datetime == "" and end_datetime == "":
                start_datetime = None
                end_datetime = None

            self.child_window = tk.Toplevel(self.master)
            self.child_window.protocol("WM_DELETE_WINDOW", self.close_Toplevel)
            child_app = ChildWindow(self.child_window, self.selected_file_path, start_datetime, end_datetime)

    def close_Toplevel(self):
        self.child_window.destroy()
        self.child_window = None




class ChildWindow: # TODO: error label. utils.py rises error, and it shows in this label
    def __init__(self, master, file_path, start_datetime, end_datetime):
        self.master = master
        self.master.geometry("1150x500")
        self.master.title("Data Visualization")

        self.data_loader = DataLoader()
        self.processor = DataProcessor()
        self.plotter = Plotter()
        self.stats = DescriptiveStats()

        self.df = self.processor.preprocess(self.data_loader.load_data(file_path, start_date=start_datetime, end_date=end_datetime))

        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        left_frame = ttk.LabelFrame(main_frame, text="Plot Selection", padding=10)
        left_frame.pack(side="left", fill="y", padx=5)

        right_frame = ttk.LabelFrame(main_frame, text="Statistics", padding=10)
        right_frame.pack(side="right", fill="both", expand=True, padx=5)

        self.add_plot_controls(left_frame)
        self.add_stats_table(right_frame)
        
    def add_plot_controls(self, frame):
        ttk.Label(frame, text="Select a time series plot:", font=("Helvetica", 10, "bold")).pack(pady=5)

        # Time series plots
        plot_buttons = [
            ("Current Timeline (IA IB IC)", ["IA", "IB", "IC"]),
            ("Power Timeline (SA SB SC)", ["SA", "SB", "SC"]),
            ("Active Energy Timeline (EPA EPB EPC)", ["EPA", "EPB", "EPC", "EPSum"]),
        ]

        for label, cols in plot_buttons:
            ttk.Button(frame, text=label, command=lambda c=cols: self.plotter.plot_time_series(self.df[['Date_Time'] + c], columns=c, title=label, min_max_arrows=self.min_max_var.get())).pack(fill='x', pady=2)                    
        
        self.min_max_var = tk.BooleanVar()
        min_max_checkbox = ttk.Checkbutton(frame, text="Show Min Max Arrows", variable=self.min_max_var)
        min_max_checkbox.pack()

        functions = [
            ("Distributions:", self.plotter.plot_gaussian_distribution_v1),
            ("Distributions (for heavy files):", self.plotter.plot_gaussian_distribution_v2)
        ]
        
        distributions = [
            ("Voltage Gaussian Dist (UA UB UC)", ['UA', 'UB', 'UC']),
            ("Current Gaussian Dist (IA IB IC)", ['IA', 'IB', 'IC']),
        ]

        for title, func in functions:
            ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=8)
            ttk.Label(frame, text=title, font=("Helvetica", 10, "bold")).pack(pady=5)
            for label, cols in distributions:
                # include cols and func in lambda to avoid late binding
                ttk.Button(frame, text=label, command=lambda c=cols, f=func: f(self.df[c], columns=c, title=label)).pack(fill='x', pady=2)
                
            # ttk.Button(frame, text="Voltage Gaussian Dist (UA UB UC)", command=lambda f=func: f(self.df[['UA', 'UB', 'UC']], columns=['UA', 'UB', 'UC'], title='Voltage Distribution')).pack(fill='x', pady=2)

    def add_stats_table(self, frame):
        # Create Two Text Widgets
        text_box_min_max = tk.Text(frame, wrap=tk.WORD, height=20, width=46)
        text_box_min_max.grid(row=0, column=0, padx=5, pady=5)

        text_box_energy = tk.Text(frame, wrap=tk.WORD, height=20, width=51)
        text_box_energy.grid(row=0, column=1, padx=5, pady=5)

        # Populate the Min/Max Text Box
        max_vals, min_vals, energy_summary = self.stats.describe(self.df)

        if not max_vals.empty and not min_vals.empty:
            min_max_text = "Stats (Min/Max):\n"
            min_max_text += "-" * 45 + "\n"  # Increased separator width

            stats_df = pd.concat([max_vals, min_vals], axis=1)
            stats_df.columns = ["Max", "Min"]

            # Adjusted column widths for better alignment
            param_width = 15  # Parameter column
            max_width = 15  # Max values column
            min_width = 15  # Min values column

            # Header row with better alignment
            min_max_text += f"{'Parameter'.ljust(param_width)}{'Max'.rjust(max_width)}{'Min'.rjust(min_width)}\n"
            min_max_text += "-" * 45 + "\n"  # Increased separator width again

            # Format each row with consistent spacing
            for index, row in stats_df.iterrows():
                # Using better number formatting with fewer decimals for large numbers
                max_val = (
                    f"{row['Max']:.2f}" if row["Max"] < 1000 else f"{row['Max']:.0f}"
                )
                min_val = (
                    f"{row['Min']:.2f}" if row["Min"] < 1000 else f"{row['Min']:.0f}"
                )

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
                energy_text += (
                    f"{row['Category'].ljust(20)}"
                    f"{row['First']:>10.0f}"
                    f"{row['Last']:>10.0f}"
                    f"{row['Change']:>10.0f}\n"
                )

            text_box_energy.insert(tk.END, energy_text)


if __name__ == "__main__":
    # root = ThemedTk(theme="arc")
    root = tk.Tk()
    # root.iconbitmap('folder_icon.ico')
    sv_ttk.set_theme("dark")  # Set theme for sv_ttk
    app = FileSelectorGUI(root)

    root.mainloop()
