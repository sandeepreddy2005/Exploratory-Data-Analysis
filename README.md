# Exploratory-Data-Analysis

This repository contains an Exploratory Data Analysis of the famous Titanic dataset using Python and popular data science libraries. The goal is to extract insights, visualize trends, and understand factors affecting survival.

---

## Dataset

The dataset used is the **Titanic dataset** available via the Seaborn library. It contains passenger information such as age, sex, class, fare, embarkation port, and survival status.

---

## Tools & Libraries Used

- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## Code Snippet (Data Cleaning & Overview)

```python
import pandas as pd
import seaborn as sns

# Load Titanic dataset
df = sns.load_dataset('titanic')

# Data overview
print(df.head())
print(df.info())
print(df.describe(include='all'))
print(df.isnull().sum())

# Data cleaning
df['age'].fillna(df['age'].median(), inplace=True)   # Fill missing ages with median
df.drop(columns=['deck'], inplace=True)              # Drop 'deck' column (too many missing values)
df.dropna(inplace=True)                              # Drop any remaining rows with missing data
