"""""
# 2023.10.20

saiku
http://193.6.5.61:8083/
cred > admin, admin

1. feladat > raiku
11.24-ig

2.feladat > python
"""

import pandas as pd
df = pd.read_csv("exam_dirtydata.csv")

# Describe data
print(df.info())

# Clean unique identifier (ID) first
# Order by given field
df.sort_values("ID", inplace = True)

# Select duplicate rows based on one column
duplicateRowsDF = df[df.duplicated(['ID'])]
print("Duplicate rows:", duplicateRowsDF, sep='\n')
# Remove duplicates from dataframe - keeping first occurrence
df.drop_duplicates(subset =['ID'], keep = "first",
                   inplace = True)
print(df.info())
# Save dataframe to file
df.to_csv("exam_data_WithoutDuplicates.csv")

# Detect missing values
# Making a list of missing value types
# Phyton will consider these values as NaN
missing_values = ["n/a", "na", "-", "--"]
df = pd.read_csv("data_WithoutDuplicates.csv",
                 na_values = missing_values)

# Total missing values for each feature
print(df.isnull().sum())

git config user.name "Szilva"


# Delete rows where Signature_date is missing
df.dropna(subset=['Signature_date'], inplace=True)
df.dropna(subset=['Grade'], inplace=True)
print(df.isnull().sum())

# Check date format
# Try to convert all cells in the 'Date' column into dates.
# If the to_datetime() fails, a NaT (Not a Time) value is
# replaced in place of the original data. Then rows with empty
# date values should be dropped.
df['Signature_date'] = pd.to_datetime(df['Signature_date'], errors='coerce')
df.dropna(subset=['Signature_date'], inplace=True)

# Check other NaN values
# Remove first Unnamed column
df.drop('Unnamed: 0', axis=1, inplace=True)
print(df.isnull().sum())

# Replace missing and incorrect Grade values using median
median = df['Grade'].median()
df['Grade'].fillna(median, inplace=True)
print(df.isnull().sum())

# Check value errors
print(df.describe())

# Visualizing data
import matplotlib.pyplot as plt
plt.style.use('ggplot')

fig, ax = plt.subplots()
# 'ID', 'Signature_date', 'Attendance', 'Written_exam', 'Oral_exam', 'Grade'
ax.boxplot((df['ID'], df['Signature_date'], df['Attendance'], df['Written_exam'], df['Oral_exam'], df['Grade']), vert=False, showmeans=True, meanline=True,
           labels=('ID', 'Signature_date', 'Attendance', 'Written_exam', 'Oral_exam', 'Grade'), patch_artist=True,
           medianprops={'linewidth': 2, 'color': 'purple'},
           meanprops={'linewidth': 2, 'color': 'red'})
plt.show()

# Check Duration with histogram
import numpy as np

hist, bin_edges = np.histogram(df['Grade'], bins=10)
fig, ax = plt.subplots()
ax.hist(df['Grade'], bin_edges, cumulative=False)
ax.set_xlabel('Grade')
ax.set_ylabel('Attendance')
plt.show()

# Replace outlier value with median
median = df['Grade'].median()
df.replace(to_replace=max(df['Grade']), value=median, inplace=True)

# Draw histogram for each numeric variable
# Normal distribution over histogram
from scipy.stats import norm

attributes = ['ID', 'Signature_date', 'Attendance', 'Written_exam', 'Written_exam', 'Grade']
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
