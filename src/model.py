from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import os

# Load data
data_folder = "data"
csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

# Initialize an empty DataFrame
data = pd.DataFrame()

# Iterate over CSV files and merge them
for file in csv_files:
    file_path = os.path.join(data_folder, file)
    if file == "pbj_hp.csv":
        df = pd.read_csv(file_path, parse_dates=["Date"], names=["Date", "PBJ_Price", "PBJ_ROC"], skiprows=1)
    elif file == "xlp_hp.csv":
        df = pd.read_csv(file_path, parse_dates=["Date"], names=["Date", "XLP_Price", "XLP_ROC"], skiprows=1)
    else:
        df = pd.read_csv(file_path, parse_dates=["Date"])
    
    # Ensure 'Date' column is datetime64[ns]
    df['Date'] = pd.to_datetime(df['Date'])
    
    if data.empty:
        data = df
    else:
        data = data.merge(df, on="Date", how="left")

# Clean the data
data = data.dropna()

# Rename columns
# Rename "Rate of Change (%)" to "Revenue_ROC", "Rate_of_Change" to "HP_ROC"
data = data.rename(columns={
    "Rate of Change (%)": "Revenue_ROC",
    "Rate_of_Change": "HP_ROC"
})

# ???Reinterpret the words "buy", sell", "hold" as numerical values???

# Display the merged DataFrame
print("LOADING DATA======================================================================================================================================================")
print(data.head())









# Feature Engineering

# Lag features
data['ANR_lag1'] = data['ANR'].shift(1)  # Previous day's ANR
data['PX_LAST_lag1'] = data['PX_LAST'].shift(1)  # Previous day's closing price
data['Revenue_lag7'] = data['Revenue'].rolling(window=7).mean()  # 7-day avg revenue

# Moving averages
data['PX_LAST_MA7'] = data['PX_LAST'].rolling(window=7).mean()  # 7-day moving average
data['ANR_MA30'] = data['ANR'].rolling(window=30).mean()  # 30-day ANR trend

# Relative performance
data['Target_Price_Gap'] = data['PX_LAST'] / data['Target Price']  # % to target
data['Undervalued'] = (data['PX_LAST'] < data['Target Price']).astype(int)  # Binary flag
data['Stock_vs_PBJ'] = data['PX_LAST'] / data['PBJ_Price']  # Relative strength to PBJ
data['Stock_vs_XLP'] = data['PX_LAST'] / data['XLP_Price']  # Relative strength to XLP

# Momentum and volatility
data['PX_ROC_5d'] = data['PX_LAST'].pct_change(5)  # 5-day price momentum
data['Revenue_ROC_30d'] = data['Revenue'].pct_change(30)

# Sentiment
data['ANR_Change_Abs'] = data['ANR Change'].abs()  # Strength of analyst sentiment shift

# Display the engineered features
print("FEATURE ENGINEERING================================================================================================================================================")
print(data.head())
print(data.columns)








# Target variable
# ???subject to change???
data['Future_5d_Return'] = data['PX_LAST'].shift(-5) / data['PX_LAST'] - 1
data['Label'] = data['Future_5d_Return'].apply(
    lambda x: 'Buy' if x > 0.015
                else 'Sell' if x < -0.015
                else 'Hold'
)
print("TARGET VARIABLE=====================================================================================================================================================")
print(data['Label'].value_counts())









# Final data preparation
data.fillna(method='ffill', inplace=True)  # Forward-fill missing values
data.dropna(inplace=True)  # Drop remaining NaN rows
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[['ANR', 'PX_LAST', 'Revenue', 'PBJ_Price']])

# Train test split to be time-series aware
train = data[data['Date'] < '2024-01-01']
test = data[data['Date'] >= '2024-01-01']

print("FINAL DATA PREPARATION=============================================================================================================================================")
print(train.head())









print("TRAINING MODEL=====================================================================================================================================================")
# Feature selection
features = [
    'ANR', 'ANR Change', 'Revenue', 'Revenue_ROC',
    'PX_LAST', 'HP_ROC', 'PBJ_Price', 'PBJ_ROC',
    'XLP_Price', 'XLP_ROC', 'ANR_lag1', 'PX_LAST_lag1',
    'Revenue_lag7', 'PX_LAST_MA7', 'ANR_MA30', 'Stock_vs_PBJ',
    'Stock_vs_XLP', 'PX_ROC_5d', 'Revenue_ROC_30d', 'ANR_Change_Abs'
]
# Excluded columns
# (a) ANR Classification
# This is your target variable (the label you’re trying to predict). Including it as a feature would cause data leakage since the model would "cheat" by seeing the answer during training. Remove it!
# (b) Target Price
# This is a forward-looking metric (analyst consensus for future price). If this is not available in real time when making predictions (e.g., analysts update it periodically), including it would leak future information.
# (c) Undervalued
# This is derived from PX_LAST and Target Price. If Target Price is excluded, Undervalued should also be excluded to avoid indirect leakage.

# Define X and y
X_train = train[features]
X_test = test[features]
y_train = train['Label']
y_test = test['Label']

# Train a model

# Encode the labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Print the mapping of labels to encoded values
print("Label encoding mapping:")
for label, encoded in zip(le.classes_, range(len(le.classes_))):
    print(f"{label}: {encoded}")

# Initialize the model
rf = RandomForestClassifier(
    n_estimators=200,  # Number of trees
    max_depth=10,       # Control overfitting
    random_state=42,    # Reproducibility
    class_weight="balanced"
)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'min_samples_split': [5, 10]  # Force splits to consider minority classes
}
grid_search = GridSearchCV(rf, param_grid, scoring='f1_macro', cv=5)
grid_search.fit(X_train, y_train_encoded)
rf = grid_search.best_estimator_

# Train the model
rf.fit(X_train, y_train_encoded)

# Predict on test data
y_pred = rf.predict(X_test)

# Decode labels back to original strings
y_pred_labels = le.inverse_transform(y_pred)

# Print metrics
print("Accuracy:", accuracy_score(y_test, y_pred_labels))
print("\nClassification Report:\n", classification_report(y_test, y_pred_labels))









# Feature Importance









# Use the model to make predictions