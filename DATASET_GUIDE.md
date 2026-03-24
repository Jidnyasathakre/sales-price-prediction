# Dataset Guide - How to Provide Your Dataset

## Method 1: Upload via Streamlit Interface (Recommended) ✨

1. **Run the application:**
   ```bash
   streamlit run app.py
   ```

2. **Upload your dataset:**
   - Look for the **"📁 Dataset"** section in the left sidebar
   - Click **"Browse files"** or drag and drop your file
   - Supported formats: **CSV**, **Excel (.xlsx, .xls)**
   - The app will automatically load your dataset

3. **Requirements:**
   - Your dataset **must have a column named 'price'** (case-sensitive) as the target variable
   - All other columns will be treated as features
   - The dataset should be clean (no missing values, or they'll be automatically removed)

## Method 2: Replace the Default File

1. **Place your CSV file:**
   - Save your dataset as `sales_data.csv`
   - Place it in the `data/` folder
   - Replace the existing file if needed

2. **Run the application:**
   ```bash
   streamlit run app.py
   ```
   - The app will automatically load `data/sales_data.csv`

## Dataset Format Requirements

### Required Column:
- **`price`** - The target variable (sales price) - **REQUIRED**

### Feature Columns:
Your dataset can have **any number of feature columns**. Common examples:
- `area_sqft` or `area` - Property area
- `bedrooms` - Number of bedrooms
- `bathrooms` - Number of bathrooms
- `age_years` or `age` - Property age
- `location_score` - Location quality score
- `garage` - Number of garage spaces
- etc.

### Data Types:
- **Numeric features**: integers or floats (e.g., area, bedrooms, age)
- **Categorical features**: integers (0/1 for binary, or numeric codes)
- **Target variable (`price`)**: numeric (integer or float)

### Example Dataset Structure:

```csv
area_sqft,bedrooms,bathrooms,age_years,garage,location_score,price
1500,3,2.5,10,2,8.5,350000
2000,4,3.0,5,2,9.0,450000
1200,2,2.0,20,1,7.0,280000
...
```

Or in Excel:
| area_sqft | bedrooms | bathrooms | age_years | garage | location_score | price |
|-----------|----------|-----------|-----------|--------|----------------|-------|
| 1500      | 3        | 2.5       | 10        | 2      | 8.5            | 350000|
| 2000      | 4        | 3.0       | 5         | 2      | 9.0            | 450000|
| 1200      | 2        | 2.0       | 20        | 1      | 7.0            | 280000|

## If Your Target Column Has a Different Name

If your target column is not named `price`, you have two options:

### Option A: Rename in Your File (Easiest)
1. Open your CSV/Excel file
2. Rename the target column to `price`
3. Save and upload

### Option B: Modify the Code
1. Open `app.py`
2. Find the line: `if 'price' not in df.columns:`
3. Replace `'price'` with your column name (e.g., `'SalePrice'`, `'price_usd'`, etc.)
4. Also update: `y = df_clean['price']` to use your column name

## Data Quality Tips

1. **Remove or handle missing values** - The app automatically removes rows with missing values
2. **Ensure numeric columns are numeric** - Not text/strings
3. **Remove any ID columns** (like `id`, `property_id`) if they're just identifiers
4. **At least 50-100 samples** recommended for meaningful model training
5. **More features = better predictions** (up to a point)

## Troubleshooting

### "Your dataset must contain a column named 'price'"
- **Solution**: Rename your target column to `price` in your file

### "Unsupported file format"
- **Solution**: Save your file as CSV or Excel (.xlsx, .xls)

### "Error loading file"
- **Solution**: 
  - Check that your file is not corrupted
  - Ensure CSV uses comma (,) as delimiter
  - Try saving as a new CSV file

### Model performs poorly
- **Possible causes**:
  - Not enough data (need more samples)
  - Features not relevant to price
  - Need feature engineering
  - Target column has errors/outliers

## Example: Preparing Your Dataset

If you have data in a different format, here's how to prepare it:

**Before (raw data):**
```csv
Property_ID,Square_Feet,Rooms,Age,List_Price
P001,1500,3,10,350000
P002,2000,4,5,450000
```

**After (prepared for app):**
```csv
area_sqft,bedrooms,age_years,price
1500,3,10,350000
2000,4,5,450000
```

Steps:
1. Remove ID column
2. Rename columns to descriptive names
3. Rename target to `price`
4. Keep only relevant features

## Need Help?

If you're unsure about your dataset format, you can:
1. Share a sample of your data (first few rows)
2. Describe what columns you have
3. I can help you prepare it or modify the code to work with your format

