import pandas as pd

df = pd.read_csv('benin-malanville.csv')


print("Summary Statistics for benin-malanville:")
print(df.describe())
