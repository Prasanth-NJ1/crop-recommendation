import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import requests
import os
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def download_dataset():
    """Download the crop recommendation dataset"""
    url = "https://raw.githubusercontent.com/Sadhana-Panthi/agricultural-crop-recommendation/main/Crop_recommendation.csv"
    
    if not os.path.exists('Crop_recommendation.csv'):
        print("Downloading dataset...")
        response = requests.get(url)
        with open('Crop_recommendation.csv', 'wb') as f:
            f.write(response.content)
        print("Dataset downloaded successfully!")
    else:
        print("Dataset already exists!")

def augment_data(df, augmentation_factor=3):
    """Augment the dataset with synthetic data"""
    print(f"Original dataset size: {len(df)}")
    
    augmented_data = []
    
    for _, row in df.iterrows():
        # Add original data
        augmented_data.append(row.to_dict())
        
        # Generate synthetic samples
        for _ in range(augmentation_factor):
            # Add small random variations to numerical features
            synthetic_row = row.copy()
            
            # Add noise to numerical columns (excluding 'label')
            numerical_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            for col in numerical_cols:
                # Add ±10% variation
                variation = np.random.uniform(-0.1, 0.1)
                synthetic_row[col] = row[col] * (1 + variation)
                
                # Ensure values stay within reasonable bounds
                if col in ['N', 'P', 'K']:
                    synthetic_row[col] = max(0, min(140, synthetic_row[col]))
                elif col == 'temperature':
                    synthetic_row[col] = max(8, min(44, synthetic_row[col]))
                elif col == 'humidity':
                    synthetic_row[col] = max(14, min(100, synthetic_row[col]))
                elif col == 'ph':
                    synthetic_row[col] = max(3.5, min(10, synthetic_row[col]))
                elif col == 'rainfall':
                    synthetic_row[col] = max(20, min(300, synthetic_row[col]))
            
            augmented_data.append(synthetic_row)
    
    augmented_df = pd.DataFrame(augmented_data)
    print(f"Augmented dataset size: {len(augmented_df)}")
    return augmented_df

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and return the best one"""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    best_model = None
    best_score = 0
    best_model_name = ""
    
    print("\nTraining and evaluating models...")
    print("-" * 50)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        cv_mean = cv_scores.mean()
        
        print(f"{name} - Test Accuracy: {accuracy:.4f}")
        print(f"{name} - CV Mean Score: {cv_mean:.4f}")
        
        # Keep track of the best model
        if cv_mean > best_score:
            best_score = cv_mean
            best_model = model
            best_model_name = name
    
    print(f"\nBest model: {best_model_name} (CV Score: {best_score:.4f})")
    return best_model, best_model_name

def main():
    # Download dataset
    download_dataset()
    
    # Load the dataset
    print("\nLoading dataset...")
    df = pd.read_csv('Crop_recommendation.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nCrop distribution:")
    print(df['label'].value_counts())
    
    # Augment the dataset
    print("\nAugmenting dataset...")
    augmented_df = augment_data(df, augmentation_factor=3)
    
    # Prepare features and target
    X = augmented_df.drop('label', axis=1)
    y = augmented_df['label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train models and get the best one
    best_model, best_model_name = train_models(X_train, X_test, y_train, y_test)
    
    # Test the best model
    y_pred = best_model.predict(X_test)
    print(f"\nFinal Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the best model
    print(f"\nSaving {best_model_name} as crop_recommendation_model.pkl...")
    with open('crop_recommendation_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    # Test the saved model
    print("\nTesting saved model...")
    sample = [[90, 42, 43, 20.5, 82.0, 6.5, 202.0]]
    predicted_crop = best_model.predict(sample)
    print(f"Sample prediction: {predicted_crop[0]}")
    
    print("\nTraining completed successfully!")
    print(f"Best model ({best_model_name}) saved as 'crop_recommendation_model.pkl'")

if __name__ == "__main__":
    main() 