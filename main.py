from utils import *

path = "./data/DataSheet_1819011001_3P4W_3.csv"
data_loader = DataLoader()
processor = DataProcessor()
plotter = Plotter()
data_stats = DescriptiveStats()

df = processor.preprocess(data_loader.load_data(path, rows=0))

# describe = data_stats.describe(df)
print(df.describe())