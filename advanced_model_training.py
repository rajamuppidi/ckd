import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report
import shap

# Load the processed dataset
df = pd.read_csv('ChronicKidneyDisease-Prediction/processed_kidney_disease.csv')

# Split the dataset into features and target variable
X = df.drop('classification', axis=1)
y = df['classification']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Perform Recursive Feature Elimination (RFE) with cross-validation
for model_name, model in models.items():
    rfe = RFE(estimator=model, n_features_to_select=10)
    rfe.fit(X_train, y_train)
    X_train_rfe = rfe.transform(X_train)
    X_test_rfe = rfe.transform(X_test)
    
    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20]
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    }
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid[model_name], cv=5, scoring='accuracy')
    grid_search.fit(X_train_rfe, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_rfe)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f'{model_name} Model (RFE + Hyperparameter Tuning):')
    print(f'Best Parameters: {grid_search.best_params_}')
    print(f'Accuracy: {accuracy:.4f}')
    print('Classification Report:')
    print(report)
    print('-' * 80)
    
    # SHAP values for feature importance
    if model_name == 'Random Forest':
        explainer = shap.Explainer(best_model, X_train_rfe)
        shap_values = explainer(X_test_rfe, check_additivity=False)
        shap.summary_plot(shap_values, X_test_rfe, feature_names=X.columns[rfe.support_])
    else:
        # Alternative method for feature importance for Gradient Boosting
        feature_importances = best_model.feature_importances_
        feature_names = X.columns[rfe.support_]
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        print('Feature Importances for Gradient Boosting:')
        print(importance_df)
