import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import numpy as np

# Load datasets
def load_data():
    dataset = pd.read_csv('dataset.txt', sep='\s+', engine='python')
    features = pd.read_csv('just_features.txt', sep='\s+', engine='python')
    return dataset, features

# Preprocess datasets
def preprocess_data(dataset, features):
    X = dataset.drop(columns=['custo'])
    y = dataset['custo']

    # Identify categorical and numerical columns
    categorical_features = ['genero', 'estado_civil', 'zona_residencia', 'fumador', 'class_etaria']
    numerical_features = ['imc']

    # Preprocessing for numeric data
    numeric_transformer = StandardScaler()

    # Preprocessing for categorical data
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Combine preprocessors in a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])

    return X, y, features, preprocessor

# Train SVM model
def train_model(X, y, preprocessor):
    # Define the model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', SVR())
    ])

    # Parameter grid for hyperparameter tuning
    param_grid = {
        'regressor__C': [0.1, 1, 10],
        'regressor__epsilon': [0.1, 0.2, 0.5],
        'regressor__kernel': ['linear', 'rbf']
    }

    # Grid search for best parameters
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
    grid_search.fit(X, y)

    return grid_search.best_estimator_

# Predict costs for new features
def predict_costs(model, features):
    predictions = model.predict(features)
    return predictions

# Main function
def main():
    # Load data
    dataset, features = load_data()

    # Preprocess data
    X, y, features_processed, preprocessor = preprocess_data(dataset, features)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = train_model(X_train, y_train, preprocessor)

    # Evaluate model
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    print(f"R2 Score: {score}")

    # Predict on new data
    predictions = predict_costs(model, features_processed)

    # Save predictions to file
    predictions_df = pd.DataFrame(predictions, columns=['custo'])
    predictions_df.to_csv('grupo#_custos_estimados.csv', index=False)

    print("Predictions saved to 'grupo#_custos_estimados.csv'")

if __name__ == '__main__':
    main()