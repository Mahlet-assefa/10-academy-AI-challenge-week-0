import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from windrose import WindroseAxes
from scipy.stats import zscore
import numpy as np





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


# Create a Windrose plot for Wind Direction and Wind Speed
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)

# Wind speed and direction
ax.bar(df['WD'], df['WS'], width=0.3, color='b', alpha=0.7, edgecolor='black')

ax.set_title('Wind Rose Plot for Wind Speed and Direction')
ax.set_xlabel('Wind Direction (°N to East)')
ax.set_ylabel('Wind Speed (m/s)')

plt.show()

# Assuming you want to show wind speed by hour
df['hour'] = df['Timestamp'].dt.hour  # Extract hour from timestamp

# Group by hour and calculate average wind speed
wind_speed_hourly = df.groupby('hour')['WS'].mean()

# Create a radial bar plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
ax.bar(wind_speed_hourly.index, wind_speed_hourly, width=0.3, color='g', alpha=0.7, edgecolor='black')

ax.set_title('Average Wind Speed by Hour')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Wind Speed (m/s)')

plt.show()

# Plot histogram for wind direction distribution
plt.figure(figsize=(8, 6))
plt.hist(df['WD'], bins=24, color='purple', edgecolor='black', alpha=0.7)
plt.title('Wind Direction Distribution')
plt.xlabel('Wind Direction (°N to East)')
plt.ylabel('Frequency')
plt.show()


# Scatter plot to visualize the relationship between RH and Tamb (Ambient Temperature)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['RH'], y=df['Tamb'], color='blue')
plt.title('Temperature vs Relative Humidity')
plt.xlabel('Relative Humidity (%)')
plt.ylabel('Ambient Temperature (°C)')
plt.show()

# Scatter plot to visualize the relationship between GHI and Tamb (Ambient Temperature)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['GHI'], y=df['Tamb'], color='red')
plt.title('Ambient Temperature vs Solar Radiation (GHI)')
plt.xlabel('Global Horizontal Irradiance (W/m²)')
plt.ylabel('Ambient Temperature (°C)')
plt.show()

# Repeat for TModA and TModB
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['GHI'], y=df['TModA'], color='green')
plt.title('Module A Temperature vs Solar Radiation (GHI)')
plt.xlabel('Global Horizontal Irradiance (W/m²)')
plt.ylabel('Module A Temperature (°C)')
plt.show()

# Correlation matrix to show the relationships between various columns
correlation_matrix = df[['Tamb', 'TModA', 'TModB', 'GHI', 'DNI', 'DHI', 'RH']].corr()

# Heatmap to visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Extract the hour from the timestamp
df['hour'] = df['Timestamp'].dt.hour

# Group by hour and calculate the average temperature for Tamb
hourly_temperature = df.groupby('hour')['Tamb'].mean()

# Plot the average temperature by hour
plt.figure(figsize=(8, 6))
hourly_temperature.plot(kind='line', color='purple', marker='o')
plt.title('Average Ambient Temperature by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Average Ambient Temperature (°C)')
plt.grid(True)
plt.show()

# Plot histogram for GHI
plt.figure(figsize=(8, 6))
sns.histplot(df['GHI'], kde=True, bins=30, color='orange')
plt.title('Histogram of Global Horizontal Irradiance (GHI)')
plt.xlabel('GHI (W/m²)')
plt.ylabel('Frequency')
plt.show()

# Plot histogram for DNI
plt.figure(figsize=(8, 6))
sns.histplot(df['DNI'], kde=True, bins=30, color='green')
plt.title('Histogram of Direct Normal Irradiance (DNI)')
plt.xlabel('DNI (W/m²)')
plt.ylabel('Frequency')
plt.show()

# Plot histogram for DHI
plt.figure(figsize=(8, 6))
sns.histplot(df['DHI'], kde=True, bins=30, color='blue')
plt.title('Histogram of Diffuse Horizontal Irradiance (DHI)')
plt.xlabel('DHI (W/m²)')
plt.ylabel('Frequency')
plt.show()

# Plot histogram for Wind Speed (WS)
plt.figure(figsize=(8, 6))
sns.histplot(df['WS'], kde=True, bins=30, color='red')
plt.title('Histogram of Wind Speed (WS)')
plt.xlabel('Wind Speed (m/s)')
plt.ylabel('Frequency')
plt.show()

# Plot histogram for Tamb (Ambient Temperature)
plt.figure(figsize=(8, 6))
sns.histplot(df['Tamb'], kde=True, bins=30, color='purple')
plt.title('Histogram of Ambient Temperature (Tamb)')
plt.xlabel('Ambient Temperature (°C)')
plt.ylabel('Frequency')
plt.show()

# Plot histogram for TModA (Module A Temperature)
plt.figure(figsize=(8, 6))
sns.histplot(df['TModA'], kde=True, bins=30, color='pink')
plt.title('Histogram of Module A Temperature (TModA)')
plt.xlabel('Module A Temperature (°C)')
plt.ylabel('Frequency')
plt.show()

# Plot histogram for TModB (Module B Temperature)
plt.figure(figsize=(8, 6))
sns.histplot(df['TModB'], kde=True, bins=30, color='cyan')
plt.title('Histogram of Module B Temperature (TModB)')
plt.xlabel('Module B Temperature (°C)')
plt.ylabel('Frequency')
plt.show()

# Calculate Z-scores for numeric columns
numeric_cols = ['GHI', 'DNI', 'DHI', 'WS', 'Tamb', 'RH', 'TModA', 'TModB', 'WSgust', 'BP', 'Precipitation']
df_zscore = df[numeric_cols].apply(zscore)

# Add Z-scores to the dataframe
df[numeric_cols + '_zscore'] = df_zscore

# Flagging outliers with Z-score > 3 or < -3
outliers = (df_zscore > 3) | (df_zscore < -3)

# Create a new column to mark outliers in each column
for col in numeric_cols:
    df[f'{col}_outlier'] = outliers[col]


# Plot boxplot for each column to identify outliers
for col in numeric_cols:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot for {col}')
    plt.xlabel(col)
    plt.show()

plt.figure(figsize=(10, 6))

# Create bubble chart
sns.scatterplot(x='GHI', y='Tamb', 
                size='RH', hue='BP', 
                sizes=(20, 200), 
                data=df, palette='viridis', legend=False, alpha=0.6)

# Add title and labels
plt.title('Bubble Chart: GHI vs Tamb with Bubble Size as RH and Color as BP')
plt.xlabel('GHI (W/m²)')
plt.ylabel('Tamb (°C)')

# Show plot
plt.show()



# Load the data
df = pd.read_csv('benin-malanville.csv')

# Summarize Data
print("Initial Data Summary:")
print(df.describe())

# Data Cleaning

# Check for missing values
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

# Handle missing values (example: fill with mean for numerical columns)
df.fillna(df.mean(), inplace=True)

# Detect and handle outliers (Z-score method)
z_scores = zscore(df.select_dtypes(include=['float64', 'int64']))
abs_z_scores = np.abs(z_scores)
outliers = (abs_z_scores > 3).all(axis=1)
df_no_outliers = df[~outliers]

# Remove negative values for specific columns
df_no_outliers = df_no_outliers[(df_no_outliers['GHI'] >= 0) & 
                                (df_no_outliers['DNI'] >= 0) & 
                                (df_no_outliers['DHI'] >= 0) & 
                                (df_no_outliers['ModA'] >= 0) & 
                                (df_no_outliers['ModB'] >= 0)]

# Ensure proper data types for time column
df_no_outliers['Timestamp'] = pd.to_datetime(df_no_outliers['Timestamp'])

# Drop unnecessary columns (e.g., Comments)
df_no_outliers.drop(columns=['Comments'], inplace=True)

# Save the cleaned data
df_no_outliers.to_csv('cleaned_dataset.csv', index=False)

# Final Summary
print("Cleaned Data Summary:")
print(df_no_outliers.describe())




