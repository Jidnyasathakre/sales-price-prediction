"""
Sales Price Prediction App with Machine Learning & Data Visualization
A comprehensive Streamlit application for predicting sales prices
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Sales Price Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def load_data(uploaded_file=None):
    """Load the dataset from uploaded file or default path"""
    if uploaded_file is not None:
        # Load from uploaded file
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format! Please upload a CSV or Excel file.")
                return None
            return df
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None
    
    # Try to load from default path
    try:
        df = pd.read_csv('data/sales_data.csv')
        return df
    except FileNotFoundError:
        return None  # Return None, error message will be shown in main()

@st.cache_data
def preprocess_data(df):
    """Preprocess the data"""
    df_clean = df.copy()
    # Remove any missing values
    df_clean = df_clean.dropna()
    return df_clean

def create_3d_scatter(df, x_col, y_col, z_col, color_col, title):
    """Create a 3D scatter plot"""
    fig = go.Figure(data=[go.Scatter3d(
        x=df[x_col],
        y=df[y_col],
        z=df[z_col],
        mode='markers',
        marker=dict(
            size=5,
            color=df[color_col],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=color_col)
        ),
        text=df.index,
        hovertemplate=f'<b>{x_col}:</b> %{{x}}<br>' +
                      f'<b>{y_col}:</b> %{{y}}<br>' +
                      f'<b>{z_col}:</b> %{{z}}<br>' +
                      f'<b>{color_col}:</b> %{{marker.color}}<extra></extra>'
    )])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col,
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=600,
        margin=dict(l=0, r=0, b=0, t=50)
    )
    return fig

def train_model(X_train, y_train, X_test, y_test, model_type):
    """Train a machine learning model"""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'SVR': SVR(kernel='rbf', C=100, gamma=0.1)
    }
    
    model = models[model_type]
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    metrics = {
        'model': model,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_test_pred': y_test_pred,
        'y_test': y_test
    }
    
    return metrics

def main():
    """Main application"""
    st.markdown('<h1 class="main-header">💰 Sales Price Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar - File Upload
    st.sidebar.title("📁 Dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your dataset (CSV or Excel)",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel file with your sales data. The file should have a 'price' column as the target variable."
    )
    
    # Load data
    df = load_data(uploaded_file=uploaded_file)
    
    if df is None:
        if uploaded_file is None:
            st.info("👆 Please upload your dataset using the file uploader in the sidebar, or place your CSV file at 'data/sales_data.csv'")
        return
    
    # Check if 'price' column exists
    if 'price' not in df.columns:
        st.error("❌ Error: Your dataset must contain a column named 'price' as the target variable.")
        st.info("Available columns in your dataset:")
        st.write(df.columns.tolist())
        st.info("💡 Tip: Rename your target column to 'price' in your CSV/Excel file, or modify the code to use your column name.")
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page",
        ["📊 Data Overview", "📈 Data Visualization", "🤖 Machine Learning Models", "🔮 Price Prediction", "📋 Model Comparison"]
    )
    
    # Preprocess data
    df_clean = preprocess_data(df)
    
    if page == "📊 Data Overview":
        st.header("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(df_clean))
        with col2:
            st.metric("Features", len(df_clean.columns) - 1)
        with col3:
            st.metric("Average Price", f"${df_clean['price'].mean():,.2f}")
        with col4:
            st.metric("Price Range", f"${df_clean['price'].min():,.0f} - ${df_clean['price'].max():,.0f}")
        
        st.subheader("Dataset Preview")
        st.dataframe(df_clean.head(20), use_container_width=True)
        
        st.subheader("Dataset Statistics")
        st.dataframe(df_clean.describe(), use_container_width=True)
        
        st.subheader("Data Types and Missing Values")
        info_df = pd.DataFrame({
            'Column': df_clean.columns,
            'Data Type': df_clean.dtypes,
            'Non-Null Count': df_clean.count(),
            'Null Count': df_clean.isnull().sum()
        })
        st.dataframe(info_df, use_container_width=True)
        
        st.subheader("Correlation Matrix")
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        corr_matrix = df_clean[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "📈 Data Visualization":
        st.header("📈 Data Visualization")
        
        # 2D Visualizations
        st.subheader("2D Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Price Distribution**")
            fig = px.histogram(df_clean, x='price', nbins=50, title="Price Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Price vs Area**")
            fig = px.scatter(df_clean, x='area_sqft', y='price', color='bedrooms', 
                           title="Price vs Area (colored by Bedrooms)")
            st.plotly_chart(fig, use_container_width=True)
        
        col3, col4 = st.columns(2)
        with col3:
            st.write("**Box Plot: Price by Bedrooms**")
            fig = px.box(df_clean, x='bedrooms', y='price', title="Price Distribution by Bedrooms")
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            st.write("**Price vs Location Score**")
            fig = px.scatter(df_clean, x='location_score', y='price', size='area_sqft',
                           title="Price vs Location Score (size = Area)")
            st.plotly_chart(fig, use_container_width=True)
        
        # 3D Visualizations
        st.subheader("🌐 3D Visualizations")
        
        col5, col6 = st.columns(2)
        
        with col5:
            st.write("**3D: Area, Bedrooms, Price**")
            fig_3d_1 = create_3d_scatter(
                df_clean, 'area_sqft', 'bedrooms', 'bathrooms', 'price',
                "3D Scatter: Area vs Bedrooms vs Bathrooms (Color = Price)"
            )
            st.plotly_chart(fig_3d_1, use_container_width=True)
        
        with col6:
            st.write("**3D: Location, Crime Rate, Price**")
            fig_3d_2 = create_3d_scatter(
                df_clean, 'location_score', 'crime_rate', 'population_density', 'price',
                "3D Scatter: Location Score vs Crime Rate vs Population Density (Color = Price)"
            )
            st.plotly_chart(fig_3d_2, use_container_width=True)
        
        # Additional 3D plot
        st.write("**3D: Age, Area, Price**")
        fig_3d_3 = create_3d_scatter(
            df_clean, 'age_years', 'area_sqft', 'location_score', 'price',
            "3D Scatter: Age vs Area vs Location Score (Color = Price)"
        )
        st.plotly_chart(fig_3d_3, use_container_width=True)
        
        # Pair plot (sample)
        st.subheader("Pair Plot (Sample)")
        sample_size = st.slider("Sample size for pair plot", 50, min(500, len(df_clean)), 200)
        sample_df = df_clean.sample(n=sample_size)
        numeric_cols_subset = ['price', 'area_sqft', 'bedrooms', 'location_score', 'age_years']
        fig = px.scatter_matrix(sample_df[numeric_cols_subset], 
                              dimensions=numeric_cols_subset,
                              color='price',
                              title="Pair Plot Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🤖 Machine Learning Models":
        st.header("🤖 Machine Learning Models")
        
        # Feature selection
        st.subheader("Feature Selection")
        feature_cols = [col for col in df_clean.columns if col != 'price']
        selected_features = st.multiselect(
            "Select features for modeling",
            feature_cols,
            default=feature_cols
        )
        
        if len(selected_features) == 0:
            st.warning("Please select at least one feature!")
            return
        
        X = df_clean[selected_features]
        y = df_clean['price']
        
        # Train-test split
        test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scaling option
        use_scaling = st.checkbox("Use feature scaling", value=True)
        if use_scaling:
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
            st.session_state['scaler'] = scaler
        
        # Model selection
        st.subheader("Model Selection & Training")
        model_types = [
            'Linear Regression',
            'Ridge Regression',
            'Lasso Regression',
            'Random Forest',
            'Gradient Boosting',
            'Decision Tree',
            'SVR'
        ]
        selected_model = st.selectbox("Select a model", model_types)
        
        if st.button("Train Model"):
            with st.spinner(f"Training {selected_model}..."):
                metrics = train_model(X_train, y_train, X_test, y_test, selected_model)
                
                st.session_state['model'] = metrics['model']
                st.session_state['model_type'] = selected_model
                st.session_state['features'] = selected_features
                
                # Display metrics
                st.subheader("Model Performance")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Test R² Score", f"{metrics['test_r2']:.4f}")
                    st.metric("Train R² Score", f"{metrics['train_r2']:.4f}")
                with col2:
                    st.metric("Test MAE", f"${metrics['test_mae']:,.2f}")
                    st.metric("Train MAE", f"${metrics['train_mae']:,.2f}")
                with col3:
                    st.metric("Test RMSE", f"${np.sqrt(metrics['test_mse']):,.2f}")
                    st.metric("CV R² (mean ± std)", f"{metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
                
                # Prediction vs Actual plot
                st.subheader("Prediction vs Actual")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_test.values,
                    y=metrics['y_test_pred'],
                    mode='markers',
                    name='Predictions',
                    marker=dict(color='blue', size=8)
                ))
                # Perfect prediction line
                min_val = min(y_test.min(), metrics['y_test_pred'].min())
                max_val = max(y_test.max(), metrics['y_test_pred'].max())
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ))
                fig.update_layout(
                    title="Predicted vs Actual Prices",
                    xaxis_title="Actual Price",
                    yaxis_title="Predicted Price",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Residual plot
                st.subheader("Residual Plot")
                residuals = y_test.values - metrics['y_test_pred']
                fig = px.scatter(
                    x=metrics['y_test_pred'],
                    y=residuals,
                    title="Residual Plot",
                    labels={'x': 'Predicted Price', 'y': 'Residuals'}
                )
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance (for tree-based models)
                if selected_model in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
                    st.subheader("Feature Importance")
                    feature_importance = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': metrics['model'].feature_importances_
                    }).sort_values('Importance', ascending=False)
                    fig = px.bar(feature_importance, x='Feature', y='Importance', 
                               title="Feature Importance")
                    st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🔮 Price Prediction":
        st.header("🔮 Price Prediction")
        
        if 'model' not in st.session_state:
            st.warning("Please train a model first in the 'Machine Learning Models' page!")
            return
        
        st.subheader("Enter Property Details")
        
        # Get feature values from user
        feature_inputs = {}
        cols = st.columns(3)
        
        feature_defaults = {
            'area_sqft': df_clean['area_sqft'].mean(),
            'bedrooms': int(df_clean['bedrooms'].mean()),
            'bathrooms': df_clean['bathrooms'].mean(),
            'age_years': int(df_clean['age_years'].mean()),
            'garage': int(df_clean['garage'].mean()),
            'location_score': df_clean['location_score'].mean(),
            'near_school': 0,
            'near_hospital': 0,
            'near_mall': 0,
            'crime_rate': df_clean['crime_rate'].mean(),
            'population_density': int(df_clean['population_density'].mean())
        }
        
        idx = 0
        for feature in st.session_state['features']:
            with cols[idx % 3]:
                if feature in ['bedrooms', 'garage', 'near_school', 'near_hospital', 'near_mall', 'population_density', 'age_years']:
                    if feature in ['near_school', 'near_hospital', 'near_mall']:
                        feature_inputs[feature] = st.selectbox(feature.replace('_', ' ').title(), [0, 1], 
                                                               index=int(feature_defaults.get(feature, 0)))
                    else:
                        feature_inputs[feature] = st.number_input(
                            feature.replace('_', ' ').title(),
                            min_value=0,
                            value=int(feature_defaults.get(feature, 0)),
                            step=1 if feature in ['bedrooms', 'garage', 'age_years'] else 100
                        )
                else:
                    feature_inputs[feature] = st.number_input(
                        feature.replace('_', ' ').title(),
                        min_value=0.0,
                        value=float(feature_defaults.get(feature, 0.0)),
                        step=0.1 if feature in ['bathrooms', 'location_score', 'crime_rate'] else 10.0
                    )
            idx += 1
        
        if st.button("Predict Price"):
            # Prepare input
            input_df = pd.DataFrame([feature_inputs])
            
            # Scale if scaler exists
            if 'scaler' in st.session_state:
                input_scaled = pd.DataFrame(
                    st.session_state['scaler'].transform(input_df),
                    columns=input_df.columns
                )
            else:
                input_scaled = input_df
            
            # Predict
            prediction = st.session_state['model'].predict(input_scaled)[0]
            
            # Display result
            st.success(f"### Predicted Price: ${prediction:,.2f}")
            
            # Show input summary
            with st.expander("View Input Summary"):
                st.dataframe(input_df.T.rename(columns={0: 'Value'}), use_container_width=True)
    
    elif page == "📋 Model Comparison":
        st.header("📋 Model Comparison")
        
        # Feature selection
        feature_cols = [col for col in df_clean.columns if col != 'price']
        selected_features = st.multiselect(
            "Select features for modeling",
            feature_cols,
            default=feature_cols,
            key="compare_features"
        )
        
        if len(selected_features) == 0:
            st.warning("Please select at least one feature!")
            return
        
        X = df_clean[selected_features]
        y = df_clean['price']
        
        # Train-test split
        test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05, key="compare_test_size")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scaling option
        use_scaling = st.checkbox("Use feature scaling", value=True, key="compare_scaling")
        if use_scaling:
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        
        model_types = [
            'Linear Regression',
            'Ridge Regression',
            'Random Forest',
            'Gradient Boosting',
            'Decision Tree'
        ]
        
        if st.button("Compare All Models"):
            with st.spinner("Training and comparing models..."):
                results = []
                
                for model_type in model_types:
                    metrics = train_model(X_train, y_train, X_test, y_test, model_type)
                    results.append({
                        'Model': model_type,
                        'Test R²': metrics['test_r2'],
                        'Test MAE': metrics['test_mae'],
                        'Test RMSE': np.sqrt(metrics['test_mse']),
                        'CV R² Mean': metrics['cv_mean'],
                        'CV R² Std': metrics['cv_std']
                    })
                
                results_df = pd.DataFrame(results)
                
                st.subheader("Comparison Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Visualization
                st.subheader("Model Performance Comparison")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(results_df, x='Model', y='Test R²', 
                               title="R² Score Comparison")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(results_df, x='Model', y='Test RMSE', 
                               title="RMSE Comparison (Lower is Better)")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Best model
                best_model = results_df.loc[results_df['Test R²'].idxmax(), 'Model']
                st.success(f"🏆 Best Model: **{best_model}** (R² = {results_df.loc[results_df['Test R²'].idxmax(), 'Test R²']:.4f})")

if __name__ == "__main__":
    main()

