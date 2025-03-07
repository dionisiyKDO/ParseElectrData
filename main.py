from utils import *

path = "./data/DataSheet_1819011001_3P4W_3.csv"
data_loader = DataLoader()
processor = DataProcessor()
plotter = Plotter()
data_stats = DescriptiveStats()

df = processor.preprocess(data_loader.load_data(path, rows=0))

# describe = data_stats.describe(df)
print(df.describe())
        # Right Frame for Statistics Display
        right_frame = ttk.Frame(paned_window)
        paned_window.add(right_frame)

        # Create a Text widget for display
        text_box = tk.Text(right_frame, wrap=tk.WORD, height=30, width=120)  # Wider text widget
        text_box.pack(pady=10, padx=10)

        # Populate the text box with descriptive statistics
        max_vals, min_vals, energy_summary = self.stats.describe(self.df)

        # Format and display statistics
        text = ""

        # Max/Min Values Section
        if not max_vals.empty and not min_vals.empty:
            text += "Descriptive Statistics:\n"
            text += "-" * 80 + "\n"

            # Combine data for aligned display
            stats_df = pd.concat([max_vals, min_vals], axis=1)
            stats_df.columns = ["Max", "Min"]

            # Determine column width dynamically
            column_width = max(len(str(val)) for val in stats_df.index) + 5
            text += f"{'Parameter'.ljust(column_width)}{'Max'.rjust(12)}{'Min'.rjust(12)}\n"
            text += "-" * (column_width + 24) + "\n"

            for index, row in stats_df.iterrows():
                text += f"{index.ljust(column_width)}{row['Max']:>12.2f}{row['Min']:>12.2f}\n"
            text += "\n"

        # Energy Changes Section
        if not energy_summary.empty:
            text += "Energy Changes:\n"
            text += "-" * (15 * 3 + 20) + "\n"

            # Determine column widths dynamically
            column_width = 15
            text += f"{'Category'.ljust(20)}{'First'.rjust(column_width)}{'Last'.rjust(column_width)}{'Change'.rjust(column_width)}\n"
            text += "-" * 65 + "\n"

            for _, row in energy_summary.iterrows():
                text += (f"{row['Category'].ljust(20)}"
                         f"{row['First']:>{column_width}.0f}"
                         f"{row['Last']:>{column_width}.0f}"
                         f"{row['Change']:>{column_width}.0f}\n")

        text_box.insert(tk.END, text)