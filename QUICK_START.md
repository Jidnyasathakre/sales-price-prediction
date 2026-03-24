# Quick Start Guide - Sales Price Prediction

## Step 1: Install Dependencies

Open PowerShell or Command Prompt in the project directory and run:

```bash
pip install -r requirements.txt
```

**Or simply double-click `setup.bat`** (Windows) - this will install packages and generate data automatically.

## Step 2: Verify Data File Exists

The sample data should already be generated at `data/sales_data.csv`. If not, run:

```bash
python generate_sample_data.py
```

## Step 3: Run the Application

Run this command:

```bash
streamlit run app.py
```

**Or simply double-click `run_app.bat`** (Windows)

## Step 4: Access the Application

After running the command:
- Your default web browser will automatically open
- If not, look for the URL in the terminal (usually: `http://localhost:8501`)
- The application will show a navigation sidebar with 5 pages

## Troubleshooting

### If you get "streamlit: command not found":
Make sure Streamlit is installed:
```bash
pip install streamlit
```

### If you get "Module not found" errors:
Install all dependencies:
```bash
pip install -r requirements.txt
```

### If data file is missing:
Generate it:
```bash
python generate_sample_data.py
```

## Application Features

Once running, you can:
1. 📊 **Data Overview** - Explore the dataset
2. 📈 **Data Visualization** - View 2D and 3D graphs
3. 🤖 **Machine Learning Models** - Train and evaluate models
4. 🔮 **Price Prediction** - Predict prices for new properties
5. 📋 **Model Comparison** - Compare all models

## Stopping the Application

Press `Ctrl+C` in the terminal to stop the Streamlit server.

