import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
df = pd.read_csv('ChronicKidneyDisease-Prediction/processed_kidney_disease_cleaned.csv')

# Display basic statistics
print(df.describe())

# Plot histograms for numerical columns
df.hist(bins=20, figsize=(20, 15))
plt.show()

# Plot correlation matrix for numeric columns only
numeric_df = df.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(15, 10))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Plot count plots for categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x=column)
    plt.show()
