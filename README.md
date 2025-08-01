# Crop Recommendation System

A machine learning system that recommends the best crop to grow based on soil and climate conditions.

## Features

- **Data Augmentation**: Expands the original dataset with synthetic data
- **Multiple Models**: Trains and compares various ML algorithms
- **Hyperparameter Tuning**: Uses GridSearchCV for optimal parameters
- **Comprehensive Evaluation**: Provides detailed performance metrics
- **REST API**: Flask-based API for predictions

## Dataset

The system uses the Crop Recommendation dataset with the following features:
- **N**: Nitrogen content in soil
- **P**: Phosphorus content in soil  
- **K**: Potassium content in soil
- **temperature**: Temperature in Celsius
- **humidity**: Relative humidity in %
- **ph**: pH value of soil
- **rainfall**: Rainfall in mm

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Training with More Data

### Option 1: Basic Training with Data Augmentation
```bash
python train_model.py
```
This script:
- Downloads the original dataset
- Augments it with synthetic data (3x expansion)
- Trains multiple models (Random Forest, Gradient Boosting, SVM, KNN)
- Saves the best performing model

### Option 2: Enhanced Training with Expanded Dataset
```bash
python collect_additional_data.py
python train_enhanced_model.py
```
This approach:
- Creates additional synthetic data based on agricultural research
- Expands dataset by ~10x (from ~2200 to ~22000 samples)
- Uses advanced hyperparameter tuning
- Provides comprehensive model evaluation

### Option 3: Step-by-Step Process

1. **Expand the dataset**:
```bash
python collect_additional_data.py
```

2. **Train the enhanced model**:
```bash
python train_enhanced_model.py
```

## Model Training Details

### Data Augmentation Techniques

1. **Synthetic Data Generation**: Creates realistic variations of existing data points
2. **Agricultural Knowledge Integration**: Uses crop-specific parameter ranges
3. **Balanced Sampling**: Ensures equal representation of all crops

### Models Trained

- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Sequential boosting algorithm
- **Extra Trees**: Extremely randomized trees
- **SVM**: Support Vector Machine with RBF kernel
- **KNN**: K-Nearest Neighbors

### Evaluation Metrics

- **Accuracy**: Overall prediction accuracy
- **Cross-validation**: 5-fold cross-validation scores
- **Classification Report**: Precision, recall, F1-score per crop
- **Feature Importance**: Most influential features

## API Usage

### Start the API Server
```bash
python app.py
```

### Make Predictions
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.5,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 202.0
  }'
```

### Python Example
```python
import requests

data = {
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.5,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 202.0
}

response = requests.post('http://localhost:5000/predict', json=data)
result = response.json()
print(f"Recommended crop: {result['recommended_crop']}")
```

## Dataset Expansion Strategy

### Original Dataset
- **Size**: ~2200 samples
- **Crops**: 22 different crops
- **Features**: 7 soil and climate parameters

### Expanded Dataset
- **Size**: ~22000 samples (10x expansion)
- **Method**: Synthetic data generation based on agricultural research
- **Quality**: Maintains realistic parameter ranges for each crop

### Data Augmentation Process

1. **Crop-Specific Ranges**: Each crop has defined parameter ranges
2. **Random Variation**: ±10% variation within realistic bounds
3. **Boundary Enforcement**: Ensures values stay within agricultural limits
4. **Balanced Generation**: Equal samples per crop type

## Model Performance

### Expected Results
- **Accuracy**: 95%+ on test set
- **Cross-validation**: 94%+ average score
- **Robustness**: Consistent performance across different soil conditions

### Model Selection
The system automatically selects the best performing model based on:
- Cross-validation scores
- Test set accuracy
- Model complexity and interpretability

## Files Description

- `app.py`: Flask API server
- `train_model.py`: Basic training with data augmentation
- `collect_additional_data.py`: Dataset expansion script
- `train_enhanced_model.py`: Advanced training with hyperparameter tuning
- `crop_recommendation_model.pkl`: Trained model file
- `model_metadata.pkl`: Training metadata and results
- `requirements.txt`: Python dependencies

## Advanced Usage

### Custom Model Training
```python
from train_enhanced_model import train_advanced_models

# Load your custom dataset
df = pd.read_csv('your_dataset.csv')
X = df.drop('label', axis=1)
y = df['label']

# Train models
best_model, model_name, results = train_advanced_models(X_train, X_test, y_train, y_test)
```

### Model Evaluation
```python
import pickle

# Load trained model
with open('crop_recommendation_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
sample = [[90, 42, 43, 20.5, 82.0, 6.5, 202.0]]
prediction = model.predict(sample)
print(f"Recommended crop: {prediction[0]}")
```

## Contributing

To improve the model:
1. Add more real agricultural data
2. Implement additional ML algorithms
3. Enhance data augmentation techniques
4. Add more crop types and parameters

## License

This project is open source and available under the MIT License.