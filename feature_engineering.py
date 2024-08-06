import pandas as pd

# Load the original dataset
original_df = pd.read_csv('ckd/kidney_disease.csv')

# Load the cleaned dataset
df = pd.read_csv('ckd/processed_kidney_disease_cleaned.csv')

# Use original age values for age group creation
df['age'] = original_df['age']

# Example of creating a new feature: Body Mass Index (BMI)
# Assuming 'weight' and 'height' columns exist in the dataset
# df['bmi'] = df['weight'] / (df['height'] / 100) ** 2

# Example of transforming an existing feature: Age groups
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 100], labels=['0-18', '19-35', '36-50', '51-65', '66+'], include_lowest=True)

# Save the dataset with new features
df.to_csv('ckd/processed_kidney_disease_with_features.csv', index=False)

# Display the dataset with new features
print(df.head())
print(df.info())
