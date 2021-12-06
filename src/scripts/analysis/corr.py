from pandas import read_csv

df = read_csv('early.csv', sep=',')

print(df['original_score'].corr(df['computed_score']))
