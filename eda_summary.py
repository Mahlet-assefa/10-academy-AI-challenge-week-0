import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


#cleaning data

# Subset the data based on the Cleaning column
cleaned_data = df[df['Cleaning'] == 1]
non_cleaned_data = df[df['Cleaning'] == 0]

# Plot ModA and ModB for cleaned vs non-cleaned data
plt.figure(figsize=(12, 6))
plt.plot(cleaned_data['Timestamp'], cleaned_data['ModA'], label='ModA (Cleaned)', color='green')
plt.plot(non_cleaned_data['Timestamp'], non_cleaned_data['ModA'], label='ModA (Non-Cleaned)', color='red', alpha=0.6)
plt.xlabel('Time')
plt.ylabel('ModA (W/m²)')
plt.title('Impact of Cleaning on ModA Sensor Readings Over Time')
plt.legend()
plt.show()

# Similarly for ModB
plt.figure(figsize=(12, 6))
plt.plot(cleaned_data['Timestamp'], cleaned_data['ModB'], label='ModB (Cleaned)', color='blue')
plt.plot(non_cleaned_data['Timestamp'], non_cleaned_data['ModB'], label='ModB (Non-Cleaned)', color='orange', alpha=0.6)
plt.xlabel('Time')
plt.ylabel('ModB (W/m²)')
plt.title('Impact of Cleaning on ModB Sensor Readings Over Time')
plt.legend()
plt.show()

# Calculate the correlation matrix for the relevant columns
correlation_columns = ['GHI', 'DNI', 'DHI', 'TModA', 'TModB', 'WS', 'WSgust', 'WD']
correlation_matrix = df[correlation_columns].corr()

# Display the correlation matrix
print(correlation_matrix)

#Plotting the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Solar Radiation, Temperature, and Wind Conditions')
plt.show()

# Create pair plot for a subset of variables
sns.pairplot(df[['GHI', 'DNI', 'DHI', 'TModA', 'TModB', 'WS', 'WSgust']])
plt.title('Pair Plot for Solar Radiation and Weather Variables')
plt.show()



