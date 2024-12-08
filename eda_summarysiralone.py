import pandas as pd

df = pd.read_csv('sierraleone-bumbuna.csv')

print("Summary Statistics for sierraleone-bumbuna:")
print(df.describe())

# look for outliers based on summary statistics
print("\nOutliers (possible based on IQR):")
for column in ['GHI', 'DNI', 'DHI', 'ModA', 'ModB', 'WS', 'WSgust']:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"{column} Outliers: {len(outliers)}")

# Check for invalid values (e.g., negative values)
print("\nInvalid Values (Negative Check):")
invalid = df[(df['GHI'] < 0) | (df['DNI'] < 0) | (df['DHI'] < 0)]
print(f"Invalid rows:\n{invalid}")
