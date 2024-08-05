
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
df = pd.read_csv('ChronicKidneyDisease-Prediction/kidney_disease.csv')

# Fix misclassifications
df['classification'] = df['classification'].str.strip()
df['classification'] = df['classification'].replace('ckd	', 'ckd')

# Remove leading/trailing spaces
columns_to_strip = ['pcv', 'wc', 'rc', 'dm', 'cad']
for column in columns_to_strip:
    df[column] = df[column].str.strip()

# Handle missing values
for column in df.columns:
    if df[column].dtype == 'object':
        df[column].fillna(df[column].mode()[0], inplace=True)
    else:
        df[column].fillna(df[column].median(), inplace=True)

# Correct label encoding
label_encoders = {}
categorical_columns = {
    'rbc': ['normal', 'abnormal'],
    'pc': ['normal', 'abnormal'],
    'pcc': ['notpresent', 'present'],
    'ba': ['notpresent', 'present'],
    'htn': ['no', 'yes'],
    'dm': ['no', 'yes'],
    'cad': ['no', 'yes'],
    'pe': ['no', 'yes'],
    'ane': ['no', 'yes'],
    'classification': ['notckd', 'ckd']
}

for column, classes in categorical_columns.items():
    le = LabelEncoder()
    le.fit(classes)
    df[column] = le.transform(df[column])
    label_encoders[column] = le

# Normalize numerical features (excluding the target variable 'classification')
scaler = StandardScaler()
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
numerical_columns = numerical_columns.drop('classification')
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Save the processed dataset
df.to_csv('ChronicKidneyDisease-Prediction/processed_kidney_disease_cleaned.csv', index=False)

# Display the processed dataset
print(df.head())
print(df.info())
