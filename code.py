import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("FIFA22.csv")

# Print available columns to verify correct naming
print("Available columns:", df.columns)

# Mapping the relevant features based on the dataset
feature_mapping = {
    "Age": "age",
    "Height": "height_cm",
    "Potential": "potential",
    "International Reputation": "international_reputation",
    "Weak Foot": "weak_foot",
    "ShotPower": "shooting",
    "ShortPassing": "passing",
    "Dribbling": "dribbling",
    "Value": "value_eur",
    "Overall": "overall",
    "Wage": "wage_eur",
    "Club Reputation": "club_reputation"
}

# Rename columns based on mapping
df = df.rename(columns=feature_mapping)

# Fix height column: Convert "177cm" → 177.0
df["height_cm"] = df["height_cm"].str.replace("cm", "").astype(float)

# Convert Value and Wage columns (€1.5M, €500K) to numeric
def convert_value(value):
    value = value.replace('€', '').replace('M', 'e6').replace('K', 'e3')
    return float(eval(value))

df['value_eur'] = df['value_eur'].apply(convert_value)
df['wage_eur'] = df['wage_eur'].apply(convert_value)

# Drop missing columns that do not exist in dataset
df = df.drop(columns=[col for col in ['club_reputation'] if col not in df.columns], errors='ignore')

df = df[list(set(feature_mapping.values()) & set(df.columns))].dropna()

# Log transformation to normalize target variable
df['value_eur'] = np.log1p(df['value_eur'])

# Split dataset into train and test sets
X = df.drop(columns=['value_eur'])
y = df['value_eur']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Gradient Boosting Model (Improved over Random Forest)
model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.03, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

# Check feature importance
feature_importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances})
print(importance_df.sort_values(by="Importance", ascending=False))

# Function to predict market value of a player
def predict_player_value(model, scaler, age, height_cm, potential, reputation, weak_foot, shooting, passing, dribbling, overall, wage_eur, club_reputation=None):
    """Predicts the market value of a football player using the trained model."""
    
    # Prepare input data
    player_data = [age, height_cm, potential, reputation, weak_foot, shooting, passing, dribbling, overall, wage_eur]
    if 'club_reputation' in X.columns:
        player_data.append(club_reputation if club_reputation is not None else 0)
    
    # Convert player attributes into a NumPy array
    player_data = np.array([player_data]).reshape(1, -1)
    
    # Scale input data using the same scaler
    player_data_scaled = scaler.transform(player_data)
    
    # Predict market value (log-transformed)
    predicted_log_value = model.predict(player_data_scaled)[0]
    
    # Convert log value back to actual value
    predicted_value = np.expm1(predicted_log_value)
    
    return round(predicted_value, 2)  # Return rounded value

# Example usage
predicted_price = predict_player_value(model, scaler, age=24, height_cm=180, potential=85, reputation=3, weak_foot=4, shooting=78, passing=85, dribbling=80, overall=85, wage_eur=200000, club_reputation=4)
print(f"Predicted Market Value: €{predicted_price}")

# Evaluate model performance
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Model R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): €{mae:.2f}")
print(f"Root Mean Square Error (RMSE): €{rmse:.2f}")

# Compare Predicted vs. Actual Values
comparison_df = pd.DataFrame({'Actual Value (€)': np.expm1(y_test), 
                              'Predicted Value (€)': np.expm1(y_pred)})
print(comparison_df.head(10))  # Show first 10 comparisons
