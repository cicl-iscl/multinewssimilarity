""" Merge data from train_csv and scrapped data into a single jsonl file."""
from src.data import CSVReader

r = CSVReader('data_dump/train_data.csv')
