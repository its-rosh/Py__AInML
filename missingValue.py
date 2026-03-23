import pandas as pd
import numpy as np

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, np.nan, 30, np.nan],
    'Salary': [50000, 60000, np.nan, 80000]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)

print("\nMissing Values Count:")
print(df.isnull().sum())

#Fill with 0
df_fill_zero = df.fillna(0)

print("\nAfter Filling with 0:")
print(df_fill_zero)

# Fill with mean 
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Salary'].fillna(df['Salary'].mean(), inplace=True)

print("\nAfter Filling with Mean:")
print(df)