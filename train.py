import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
df = pd.read_csv(url)

print("Data loaded successfully")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")


print("Training Model 1: Linear Regression")
X1 = df[['100g_USD']].dropna()
y1 = df.loc[X1.index, 'rating']

model_1 = LinearRegression()
model_1.fit(X1, y1)

# Save model_1
with open('model_1.pickle', 'wb') as f:
    pickle.dump(model_1, f)
print("Model 1 saved as model_1.pickle")


print("Training Model 2: Decision Tree Regressor ")

# Get data with both features
df_model2 = df[['100g_USD', 'roast', 'rating']].dropna()

# Create roast category mapping
unique_roasts = df_model2['roast'].unique()
roast_cat = {roast: idx for idx, roast in enumerate(sorted(unique_roasts))}
print(f"Roast categories: {roast_cat}")

# Convert roast to numerical labels
df_model2['roast_encoded'] = df_model2['roast'].map(roast_cat)

# Prepare features and target
X2 = df_model2[['100g_USD', 'roast_encoded']]
y2 = df_model2['rating']

# Train Decision Tree Regressor
model_2 = DecisionTreeRegressor(random_state=42)
model_2.fit(X2, y2)

# Save model_2 and roast_cat dictionary
with open('model_2.pickle', 'wb') as f:
    pickle.dump({'model': model_2, 'roast_cat': roast_cat}, f)
print("Model 2 and roast categories saved as model_2.pickle")

print("Training Complete")
print("Models saved:")
print("  - model_1.pickle: Linear Regression (100g_USD only)")
print("  - model_2.pickle: Decision Tree Regressor (100g_USD + roast)")
