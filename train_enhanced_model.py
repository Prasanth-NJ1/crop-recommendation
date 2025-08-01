import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
import requests
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def download_and_prepare_data():
    """Download dataset and prepare expanded version"""
    url = "https://raw.githubusercontent.com/Sadhana-Panthi/agricultural-crop-recommendation/main/Crop_recommendation.csv"
    
    if not os.path.exists('Crop_recommendation.csv'):
        print("Downloading dataset...")
        response = requests.get(url)
        with open('Crop_recommendation.csv', 'wb') as f:
            f.write(response.content)
        print("Dataset downloaded successfully!")
    
    # Load original data
    original_df = pd.read_csv('Crop_recommendation.csv')
    print(f"Original dataset size: {len(original_df)}")
    
    # Create expanded dataset if it doesn't exist
    if not os.path.exists('expanded_crop_recommendation.csv'):
        print("Creating expanded dataset...")
        from collect_additional_data import merge_datasets
        expanded_df = merge_datasets()
    else:
        expanded_df = pd.read_csv('expanded_crop_recommendation.csv')
        print(f"Expanded dataset loaded: {len(expanded_df)} samples")
    
    return expanded_df

def analyze_data(df):
    """Analyze the dataset characteristics"""
    print("\n=== Dataset Analysis ===")
    print(f"Total samples: {len(df)}")
    print(f"Features: {list(df.columns[:-1])}")
    print(f"Target variable: {df.columns[-1]}")
    
    print("\nCrop distribution:")
    crop_counts = df['label'].value_counts()
    print(crop_counts)
    
    print("\nFeature statistics:")
    print(df.describe())
    
    # Correlation analysis (only numerical features)
    print("\nFeature correlations:")
    numerical_df = df.drop('label', axis=1)
    correlation_matrix = numerical_df.corr()
    print(correlation_matrix)

def train_advanced_models(X_train, X_test, y_train, y_test):
    """Train multiple advanced models with hyperparameter tuning"""
    
    # Define models with parameter grids for tuning
    models = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        },
        'Extra Trees': {
            'model': ExtraTreesClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'SVM': {
            'model': SVC(random_state=42),
            'params': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'poly']
            }
        }
    }
    
    best_model = None
    best_score = 0
    best_model_name = ""
    results = {}
    
    print("\n=== Model Training and Evaluation ===")
    print("=" * 60)
    
    for name, model_info in models.items():
        print(f"\nTraining {name}...")
        
        # Grid search for hyperparameter tuning
        grid_search = GridSearchCV(
            model_info['model'], 
            model_info['params'], 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model_cv = grid_search.best_estimator_
        
        # Evaluate on test set
        y_pred = best_model_cv.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(best_model_cv, X_train, y_train, cv=5)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"{name} Results:")
        print(f"  Best parameters: {grid_search.best_params_}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(f"  CV Mean Score: {cv_mean:.4f} (+/- {cv_std:.4f})")
        
        results[name] = {
            'model': best_model_cv,
            'test_accuracy': test_accuracy,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'predictions': y_pred
        }
        
        # Keep track of the best model
        if cv_mean > best_score:
            best_score = cv_mean
            best_model = best_model_cv
            best_model_name = name
    
    print(f"\n=== Best Model Summary ===")
    print(f"Best model: {best_model_name}")
    print(f"CV Score: {best_score:.4f}")
    
    return best_model, best_model_name, results

def evaluate_model_performance(model, X_test, y_test, model_name):
    """Comprehensive model evaluation"""
    y_pred = model.predict(X_test)
    
    print(f"\n=== {model_name} Performance Analysis ===")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix Shape: {cm.shape}")
    
    # Feature importance (for tree-based models)
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        feature_names = X_test.columns if hasattr(X_test, 'columns') else [f'Feature_{i}' for i in range(len(feature_importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10))

def save_model_and_metadata(model, model_name, results, X_train, y_train):
    """Save the model and training metadata"""
    
    # Save the model
    with open('crop_recommendation_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save training metadata
    metadata = {
        'model_name': model_name,
        'training_samples': len(X_train),
        'test_samples': len(X_train) // 4,  # Assuming 80-20 split
        'feature_names': list(X_train.columns) if hasattr(X_train, 'columns') else [f'Feature_{i}' for i in range(X_train.shape[1])],
        'model_results': results,
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('model_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\nModel saved as 'crop_recommendation_model.pkl'")
    print(f"Metadata saved as 'model_metadata.pkl'")

def main():
    print("=== Enhanced Crop Recommendation Model Training ===")
    
    # Download and prepare data
    df = download_and_prepare_data()
    
    # Analyze the data
    analyze_data(df)
    
    # Prepare features and target
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train models and get the best one
    best_model, best_model_name, results = train_advanced_models(X_train, X_test, y_train, y_test)
    
    # Evaluate the best model
    evaluate_model_performance(best_model, X_test, y_test, best_model_name)
    
    # Save model and metadata
    save_model_and_metadata(best_model, best_model_name, results, X_train, y_train)
    
    # Test the saved model
    print("\n=== Model Testing ===")
    sample = [[90, 42, 43, 20.5, 82.0, 6.5, 202.0]]
    predicted_crop = best_model.predict(sample)
    print(f"Sample prediction: {predicted_crop[0]}")
    
    print("\n=== Training Completed Successfully! ===")
    print(f"Best model: {best_model_name}")
    print("Model ready for deployment!")

if __name__ == "__main__":
    main() 