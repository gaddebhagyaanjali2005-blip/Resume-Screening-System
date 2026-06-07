import pandas as pd

# Load Dataset
df = pd.read_csv("dataset/clean_resume_data.csv")

# Display first 5 rows
print(df.head())

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumns:")
print(df.columns)

# Number of categories
print("\nNumber of Categories:")
print(df["Category"].nunique())