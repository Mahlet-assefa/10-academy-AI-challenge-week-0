import pandas as pd

df = pd.read_csv('togo-dapaong_qc.csv')

print("Summary Statistics for togo:")
print(df.describe())
