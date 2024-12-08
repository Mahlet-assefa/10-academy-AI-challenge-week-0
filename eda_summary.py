import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('benin-malanville.csv')


print("Summary Statistics for benin-malanville:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

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

# Convert the 'Timestamp' column to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Plot GHI, DNI, DHI over time

print(df.columns)

plt.figure(figsize=(10, 6))
plt.plot(df['Timestamp'], df['GHI'], label='GHI', color='blue')
plt.plot(df['Timestamp'], df['DNI'], label='DNI', color='orange')
plt.plot(df['Timestamp'], df['DHI'], label='DHI', color='green')
plt.xlabel('Time')
plt.ylabel('Irradiance (W/m²)')
plt.title('Solar Irradiance Over Time')
plt.legend()
plt.show()

# Plot Tamb over time
plt.figure(figsize=(10, 6))
plt.plot(df['Timestamp'], df['Tamb'], label='Ambient Temperature (Tamb)', color='red')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.title('Ambient Temperature Over Time')
plt.legend()
plt.show()

