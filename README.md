# Sales Price Prediction System

A comprehensive machine learning and data visualization application for predicting sales prices using Streamlit.

## Features

- 📊 **Data Exploration**: Complete dataset overview with statistics and correlation analysis
- 📈 **Data Visualization**: 
  - 2D visualizations (histograms, scatter plots, box plots)
  - 3D interactive graphs using Plotly
  - Pair plots for feature relationships
- 🤖 **Machine Learning Models**:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Decision Tree Regressor
  - Support Vector Regressor (SVR)
- 🔮 **Price Prediction**: Interactive interface to predict prices for new properties
- 📋 **Model Comparison**: Compare multiple models side-by-side

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Generate sample data (if not already generated):
```bash
python generate_sample_data.py
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser.

## Dataset

The sample dataset includes the following features:
- `area_sqft`: Property area in square feet
- `bedrooms`: Number of bedrooms
- `bathrooms`: Number of bathrooms
- `age_years`: Age of the property
- `garage`: Number of garage spaces
- `location_score`: Location quality score (1-10)
- `near_school`: Binary indicator (0 or 1)
- `near_hospital`: Binary indicator (0 or 1)
- `near_mall`: Binary indicator (0 or 1)
- `crime_rate`: Crime rate in the area (1-10)
- `population_density`: Population density
- `price`: Target variable (sales price)

## Application Pages

1. **Data Overview**: View dataset statistics, preview, and correlation matrix
2. **Data Visualization**: Explore data with 2D and 3D interactive graphs
3. **Machine Learning Models**: Train and evaluate different ML models
4. **Price Prediction**: Predict prices for new properties
5. **Model Comparison**: Compare performance of multiple models

## Technologies Used

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **Plotly**: Interactive 3D visualizations
- **Matplotlib & Seaborn**: Additional plotting capabilities

## License

This project is open source and available for educational purposes.

