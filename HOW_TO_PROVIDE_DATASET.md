# How to Provide Your Dataset

## 🎯 Quick Answer: Three Easy Ways

### Method 1: Upload via App (Easiest!) ⭐
1. Run the app: `streamlit run app.py`
2. Look for **"📁 Dataset"** in the left sidebar
3. Click **"Browse files"** and select your CSV or Excel file
4. Done! The app will automatically load your data

### Method 2: Replace Default File
1. Save your dataset as `sales_data.csv`
2. Put it in the `data/` folder (replace existing file)
3. Run the app - it will load automatically

### Method 3: Share with Me
Just tell me:
- What format is your data? (CSV, Excel, etc.)
- What columns does it have?
- What's your target column name? (currently expects `price`)
- I can modify the code to work with your format

## 📋 Dataset Requirements

**Required:**
- ✅ Must have a column named **`price`** (the target variable to predict)
- ✅ File format: CSV (.csv) or Excel (.xlsx, .xls)
- ✅ All other columns will be treated as features

**Example format:**
```csv
area_sqft,bedrooms,bathrooms,age_years,location_score,price
1500,3,2.5,10,8.5,350000
2000,4,3.0,5,9.0,450000
```

**That's it!** The app will automatically:
- Detect your features
- Handle missing values
- Train models on your data
- Create visualizations

See `DATASET_GUIDE.md` for detailed instructions.

