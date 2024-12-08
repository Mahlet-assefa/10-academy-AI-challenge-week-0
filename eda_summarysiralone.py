import pandas as pd

df = pd.read_csv('sierraleone-bumbuna.csv')

print("Summary Statistics for sierraleone-bumbuna:")
print(df.describe())
