"""
Script to generate sample sales price prediction dataset
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data
n_samples = 1000

# Feature generation
data = {
    'area_sqft': np.random.randint(500, 5000, n_samples),
    'bedrooms': np.random.randint(1, 6, n_samples),
    'bathrooms': np.random.uniform(1, 4, n_samples).round(1),
    'age_years': np.random.randint(0, 50, n_samples),
    'garage': np.random.randint(0, 3, n_samples),
    'location_score': np.random.uniform(1, 10, n_samples).round(2),
    'near_school': np.random.choice([0, 1], n_samples),
    'near_hospital': np.random.choice([0, 1], n_samples),
    'near_mall': np.random.choice([0, 1], n_samples),
    'crime_rate': np.random.uniform(1, 10, n_samples).round(2),
    'population_density': np.random.randint(100, 10000, n_samples),
}

# Create price based on features with some randomness
data['price'] = (
    data['area_sqft'] * 150 +
    data['bedrooms'] * 25000 +
    data['bathrooms'] * 30000 -
    data['age_years'] * 2000 +
    data['garage'] * 15000 +
    data['location_score'] * 50000 +
    data['near_school'] * 30000 +
    data['near_hospital'] * 25000 +
    data['near_mall'] * 20000 -
    data['crime_rate'] * 5000 +
    data['population_density'] * 5 +
    np.random.normal(0, 50000, n_samples)
)

# Ensure price is positive
data['price'] = np.maximum(data['price'], 50000)

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('data/sales_data.csv', index=False)
print(f"Generated {n_samples} samples and saved to data/sales_data.csv")
print(f"Price range: ${df['price'].min():,.2f} - ${df['price'].max():,.2f}")

