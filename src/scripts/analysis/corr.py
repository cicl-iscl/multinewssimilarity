from pandas import read_csv

from src.config import SCORE_CSV

df = read_csv(SCORE_CSV, sep=',')

print(df['original_score'].corr(df['computed_score']))
