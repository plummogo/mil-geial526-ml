"""
Data cleaning on dirtydata.csv
Data description:
A person records his trainings per day:
Duration,Date,Pulse,Maxpulse,Calories
"""

# Read data from file into dataframe
import pandas as pd
df = pd.read_csv("dirtydata.csv")

# Describe data
print(df.info())

# Clean unique identifier (date) first
# Order by given field
df.sort_values("Date", inplace = True)

# Select duplicate rows based on one column
duplicateRowsDF = df[df.duplicated(['Date'])]
print("Duplicate rows:", duplicateRowsDF, sep='\n')
# Remove duplicates from dataframe - keeping first occurrence
df.drop_duplicates(subset =['Date'], keep = "first",
                   inplace = True)
print(df.info())

# Save dataframe to file
df.to_csv("data_WithoutDuplicates.csv")

# Detect missing values
# Making a list of missing value types
# Phyton will consider these values as NaN
missing_values = ["n/a", "na", "-", "--"]
df = pd.read_csv("data_WithoutDuplicates.csv",
                 na_values = missing_values)

# Total missing values for each feature
print(df.isnull().sum())

# Delete rows where Date is missing
df.dropna(subset=['Date'], inplace=True)
print(df.isnull().sum())

# Check date format
# Try to convert all cells in the 'Date' column into dates.
# If the to_datetime() fails, a NaT (Not a Time) value is
# replaced in place of the original data. Then rows with empty
# date values should be dropped.
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.dropna(subset=['Date'], inplace=True)

# Check other NaN values
# Remove first Unnamed column
df.drop('Unnamed: 0', axis=1, inplace=True)
print(df.isnull().sum())

# Replace missing and incorrect CALORIES values using median
median = df['Calories'].median()
df['Calories'].fillna(median, inplace=True)
print(df.isnull().sum())

# Check value errors
print(df.describe())

# Visualizing data
import matplotlib.pyplot as plt
plt.style.use('ggplot')

fig, ax = plt.subplots()
ax.boxplot((df['Duration'], df['Pulse'], df['Maxpulse'], df['Calories']), vert=False, showmeans=True, meanline=True,
           labels=('Duration', 'Pulse', 'Maxpulse', 'Calories'), patch_artist=True,
           medianprops={'linewidth': 2, 'color': 'purple'},
           meanprops={'linewidth': 2, 'color': 'red'})
plt.show()

# Check Duration with histogram
import numpy as np

hist, bin_edges = np.histogram(df['Duration'], bins=10)
fig, ax = plt.subplots()
ax.hist(df['Duration'], bin_edges, cumulative=False)
ax.set_xlabel('Duration')
ax.set_ylabel('Frequency')
plt.show()

# Replace outlier value with median
median = df['Duration'].median()
df.replace(to_replace=max(df['Duration']), value=median, inplace=True)

# Draw histogram for each numeric variable
# Normal distribution over histogram
from scipy.stats import norm

attributes = ['Duration', 'Pulse', 'Maxpulse', 'Calories']
for attr in attributes:
    # Fit a normal distribution to the data:
    # mean and standard deviation
    mu, std = norm.fit(df[attr])
    # Plot the histogram
    plt.hist(df[attr], bins=10, density=True, alpha=0.6, color='b')
    # Plot density function over histogram
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit Values: {:.2f} and {:.2f}".format(mu, std)
    plt.title(attr)
    plt.show()

# Save dataframe to file
df.to_csv("data_cleaned.csv")
