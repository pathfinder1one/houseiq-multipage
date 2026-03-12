"""
HouseFind — Competition-Grade Random Forest Training
Dataset: 12,000 rows × 14 features
Run once: python train_model.py
"""
import numpy as np
import pandas as pd
import pickle, os, json

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)
N = 12000

# ── Simulate realistic dataset ──────────────────────────────────────────────
cities = ['New York','Los Angeles','Chicago','Houston','Phoenix','Philadelphia','San Antonio','San Diego','Dallas','Austin']
city_idx      = np.random.randint(0, len(cities), N)
city_mult     = np.array([2.2, 2.0, 1.5, 1.1, 1.0, 1.4, 1.0, 1.8, 1.2, 1.6])

area_sqft     = np.random.randint(400, 7500, N).astype(float)
bedrooms      = np.random.randint(1, 8, N).astype(float)
bathrooms     = np.random.randint(1, 6, N).astype(float)
floors        = np.random.randint(1, 5, N).astype(float)
age_years     = np.random.randint(0, 80, N).astype(float)
garage_cars   = np.random.randint(0, 5, N).astype(float)
has_pool      = np.random.randint(0, 2, N).astype(float)
has_garden    = np.random.randint(0, 2, N).astype(float)
location_type = np.random.choice([0, 1, 2], N, p=[0.3, 0.45, 0.25]).astype(float)  # 0=rural,1=suburb,2=city
condition     = np.random.randint(1, 6, N).astype(float)
school_dist   = np.round(np.random.uniform(0.1, 12.0, N), 1)
crime_index   = np.round(np.random.uniform(0.5, 10.0, N), 1)
renovated     = np.random.randint(0, 2, N).astype(float)
lot_size      = np.random.randint(1000, 20000, N).astype(float)

# Realistic price formula
base = (
    area_sqft      * 130
    + bedrooms     * 14000
    + bathrooms    * 11500
    + floors       * 7500
    - age_years    * 1100
    + garage_cars  * 9500
    + has_pool     * 28000
    + has_garden   * 12000
    + location_type* 45000
    + condition    * 16000
    - school_dist  * 2500
    - crime_index  * 4500
    + renovated    * 22000
    + lot_size     * 4
)

price = base * city_mult[city_idx] + np.random.normal(0, 18000, N)
price = np.clip(price, 35000, 4500000).astype(int)

df = pd.DataFrame({
    'area_sqft':    area_sqft,
    'bedrooms':     bedrooms,
    'bathrooms':    bathrooms,
    'floors':       floors,
    'age_years':    age_years,
    'garage_cars':  garage_cars,
    'has_pool':     has_pool,
    'has_garden':   has_garden,
    'location_type':location_type,
    'condition':    condition,
    'school_dist_km': school_dist,
    'crime_index':  crime_index,
    'renovated':    renovated,
    'lot_size_sqft':lot_size,
    'city_idx':     city_idx,
    'price':        price,
})

print(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(df.describe().to_string())

FEATURES = ['area_sqft','bedrooms','bathrooms','floors','age_years','garage_cars',
            'has_pool','has_garden','location_type','condition','school_dist_km',
            'crime_index','renovated','lot_size_sqft','city_idx']

X = df[FEATURES]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Ensemble model ──────────────────────────────────────────────────────────
rf = RandomForestRegressor(n_estimators=350, max_depth=22, min_samples_split=3,
                           min_samples_leaf=2, max_features='sqrt', n_jobs=-1, random_state=42)

gb = GradientBoostingRegressor(n_estimators=250, learning_rate=0.07, max_depth=6,
                               subsample=0.85, min_samples_split=4, random_state=42)

ensemble = VotingRegressor([('rf', rf), ('gb', gb)], weights=[0.55, 0.45])

print("\n⏳ Training ensemble (RF + GradientBoosting) on 12,000 samples...")
ensemble.fit(X_train, y_train)

y_pred  = ensemble.predict(X_test)
mae     = mean_absolute_error(y_test, y_pred)
rmse    = np.sqrt(mean_squared_error(y_test, y_pred))
r2      = r2_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"  R²   : {r2:.4f} ({r2*100:.2f}%)")
print(f"  MAE  : ${mae:,.0f}")
print(f"  RMSE : ${rmse:,.0f}")
print(f"{'='*50}")

# Fit RF alone for importances
rf.fit(X_train, y_train)
importances = {f: float(v) for f, v in zip(FEATURES, rf.feature_importances_)}

# Price range stats per city
city_stats = {}
for i, city in enumerate(cities):
    mask = df['city_idx'] == i
    city_stats[city] = {
        'avg':    int(df.loc[mask,'price'].mean()),
        'median': int(df.loc[mask,'price'].median()),
        'min':    int(df.loc[mask,'price'].min()),
        'max':    int(df.loc[mask,'price'].max()),
        'count':  int(mask.sum()),
    }

# Monthly trend simulation
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
base_p  = 380000
trend   = [int(base_p + i*4200 + np.random.randint(-8000, 8000)) for i in range(12)]

os.makedirs('model', exist_ok=True)
payload = {
    'model':        ensemble,
    'features':     FEATURES,
    'cities':       cities,
    'city_stats':   city_stats,
    'monthly_trend':dict(zip(months, trend)),
    'dataset_stats':{
        'rows':        N,
        'features':    len(FEATURES),
        'price_min':   int(df['price'].min()),
        'price_max':   int(df['price'].max()),
        'price_mean':  int(df['price'].mean()),
        'price_median':int(df['price'].median()),
    },
    'metrics': {'r2': round(r2,4), 'mae': round(mae,2), 'rmse': round(rmse,2)},
    'feature_importances': importances,
}

with open('model/house_price_model.pkl','wb') as f:
    pickle.dump(payload, f)

print("\n✅ Model saved → model/house_price_model.pkl")
