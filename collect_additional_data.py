import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import random

def create_synthetic_crop_data():
    """Create additional synthetic crop data based on agricultural knowledge"""
    
    # Define crop-specific parameter ranges based on agricultural research
    crop_parameters = {
        'rice': {
            'N': (80, 120), 'P': (40, 60), 'K': (40, 60),
            'temperature': (22, 32), 'humidity': (70, 90), 'ph': (5.5, 7.5), 'rainfall': (100, 200)
        },
        'maize': {
            'N': (90, 130), 'P': (50, 70), 'K': (50, 70),
            'temperature': (18, 32), 'humidity': (60, 80), 'ph': (5.5, 7.5), 'rainfall': (80, 150)
        },
        'chickpea': {
            'N': (20, 40), 'P': (40, 60), 'K': (20, 40),
            'temperature': (20, 30), 'humidity': (50, 70), 'ph': (6.0, 7.5), 'rainfall': (60, 120)
        },
        'kidneybeans': {
            'N': (30, 50), 'P': (40, 60), 'K': (30, 50),
            'temperature': (20, 30), 'humidity': (60, 80), 'ph': (6.0, 7.5), 'rainfall': (80, 150)
        },
        'pigeonpeas': {
            'N': (20, 40), 'P': (30, 50), 'K': (20, 40),
            'temperature': (25, 35), 'humidity': (60, 80), 'ph': (6.0, 7.5), 'rainfall': (80, 150)
        },
        'mothbeans': {
            'N': (20, 40), 'P': (30, 50), 'K': (20, 40),
            'temperature': (25, 35), 'humidity': (50, 70), 'ph': (6.0, 7.5), 'rainfall': (60, 120)
        },
        'mungbean': {
            'N': (20, 40), 'P': (30, 50), 'K': (20, 40),
            'temperature': (25, 35), 'humidity': (60, 80), 'ph': (6.0, 7.5), 'rainfall': (80, 150)
        },
        'blackgram': {
            'N': (20, 40), 'P': (30, 50), 'K': (20, 40),
            'temperature': (25, 35), 'humidity': (60, 80), 'ph': (6.0, 7.5), 'rainfall': (80, 150)
        },
        'lentil': {
            'N': (20, 40), 'P': (30, 50), 'K': (20, 40),
            'temperature': (15, 25), 'humidity': (50, 70), 'ph': (6.0, 7.5), 'rainfall': (60, 120)
        },
        'pomegranate': {
            'N': (60, 100), 'P': (40, 60), 'K': (40, 60),
            'temperature': (20, 35), 'humidity': (60, 80), 'ph': (6.0, 7.5), 'rainfall': (80, 150)
        },
        'banana': {
            'N': (80, 120), 'P': (40, 60), 'K': (60, 100),
            'temperature': (25, 35), 'humidity': (70, 90), 'ph': (5.5, 7.5), 'rainfall': (100, 200)
        },
        'mango': {
            'N': (60, 100), 'P': (40, 60), 'K': (40, 60),
            'temperature': (25, 35), 'humidity': (60, 80), 'ph': (6.0, 7.5), 'rainfall': (80, 150)
        },
        'grapes': {
            'N': (60, 100), 'P': (40, 60), 'K': (40, 60),
            'temperature': (20, 30), 'humidity': (60, 80), 'ph': (6.0, 7.5), 'rainfall': (80, 150)
        },
        'watermelon': {
            'N': (60, 100), 'P': (40, 60), 'K': (40, 60),
            'temperature': (25, 35), 'humidity': (60, 80), 'ph': (6.0, 7.5), 'rainfall': (80, 150)
        },
        'muskmelon': {
            'N': (60, 100), 'P': (40, 60), 'K': (40, 60),
            'temperature': (25, 35), 'humidity': (60, 80), 'ph': (6.0, 7.5), 'rainfall': (80, 150)
        },
        'apple': {
            'N': (60, 100), 'P': (40, 60), 'K': (40, 60),
            'temperature': (15, 25), 'humidity': (60, 80), 'ph': (6.0, 7.5), 'rainfall': (80, 150)
        },
        'orange': {
            'N': (60, 100), 'P': (40, 60), 'K': (40, 60),
            'temperature': (20, 30), 'humidity': (60, 80), 'ph': (6.0, 7.5), 'rainfall': (80, 150)
        },
        'papaya': {
            'N': (60, 100), 'P': (40, 60), 'K': (40, 60),
            'temperature': (25, 35), 'humidity': (70, 90), 'ph': (6.0, 7.5), 'rainfall': (100, 200)
        },
        'coconut': {
            'N': (60, 100), 'P': (40, 60), 'K': (40, 60),
            'temperature': (25, 35), 'humidity': (70, 90), 'ph': (6.0, 7.5), 'rainfall': (100, 200)
        },
        'cotton': {
            'N': (80, 120), 'P': (40, 60), 'K': (40, 60),
            'temperature': (25, 35), 'humidity': (60, 80), 'ph': (6.0, 7.5), 'rainfall': (80, 150)
        },
        'jute': {
            'N': (60, 100), 'P': (40, 60), 'K': (40, 60),
            'temperature': (25, 35), 'humidity': (70, 90), 'ph': (6.0, 7.5), 'rainfall': (100, 200)
        },
        'coffee': {
            'N': (60, 100), 'P': (40, 60), 'K': (40, 60),
            'temperature': (20, 30), 'humidity': (70, 90), 'ph': (6.0, 7.5), 'rainfall': (100, 200)
        }
    }
    
    additional_data = []
    
    for crop, params in crop_parameters.items():
        # Generate 50 samples per crop
        for _ in range(50):
            sample = {}
            for param, (min_val, max_val) in params.items():
                # Generate random value within the range
                sample[param] = np.random.uniform(min_val, max_val)
            sample['label'] = crop
            additional_data.append(sample)
    
    return pd.DataFrame(additional_data)

def merge_datasets():
    """Merge original dataset with additional synthetic data"""
    
    # Load original dataset
    original_df = pd.read_csv('Crop_recommendation.csv')
    print(f"Original dataset size: {len(original_df)}")
    
    # Create additional synthetic data
    additional_df = create_synthetic_crop_data()
    print(f"Additional synthetic data size: {len(additional_df)}")
    
    # Merge datasets
    merged_df = pd.concat([original_df, additional_df], ignore_index=True)
    print(f"Combined dataset size: {len(merged_df)}")
    
    # Shuffle the data
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save the expanded dataset
    merged_df.to_csv('expanded_crop_recommendation.csv', index=False)
    print("Expanded dataset saved as 'expanded_crop_recommendation.csv'")
    
    # Show crop distribution
    print("\nCrop distribution in expanded dataset:")
    print(merged_df['label'].value_counts())
    
    return merged_df

if __name__ == "__main__":
    print("Creating additional synthetic crop data...")
    expanded_df = merge_datasets()
    print("\nDataset expansion completed!") 