from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
    elif file == "CONCCONF_HP.csv":
        df = pd.read_csv(file_path, parse_dates=["Date"], names=["Date", "CONCCONF_Price", "CONCCONF_ROC"], skiprows=1) # take out
    elif file == "M2_HP.csv":
        df = pd.read_csv(file_path, parse_dates=["Date"], names=["Date", "M2_Price", "M2_ROC"], skiprows=1) # take out
    elif file == "PCUSEQTR_HP.csv":
        df = pd.read_csv(file_path, parse_dates=["Date"], names=["Date", "PCUSEQTR_Price", "PCUSEQTR_ROC"], skiprows=1)
    elif file == "VIX_HP.csv":
        df = pd.read_csv(file_path, parse_dates=["Date"], names=["Date", "VIX_Price", "VIX_ROC"], skiprows=1)
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
print(data.columns)
# count the rows
print(data.shape)








# ================================================================================================
# FEATURE ENGINEERING
# ================================================================================================

# Lag features
data['ANR_lag1'] = data['ANR'].shift(1)
data['PX_LAST_lag1'] = data['PX_LAST'].shift(1)
data['Revenue_lag9'] = data['Revenue'].rolling(window=9, min_periods=1).mean()  # Added min_periods

# Moving averages
data['PX_LAST_MA9'] = data['PX_LAST'].rolling(window=9, min_periods=1).mean()
data['ANR_MA20'] = data['ANR'].rolling(window=20, min_periods=1).mean()

# Relative performance (with zero-division handling)
data['Target_Price_Gap'] = np.where(
    data['Target Price'] != 0,
    data['PX_LAST'] / data['Target Price'],
    np.nan
)
data['Undervalued'] = (data['PX_LAST'] < data['Target Price']).astype(int)
data['Stock_vs_PBJ'] = np.where(
    data['PBJ_Price'] != 0,
    data['PX_LAST'] / data['PBJ_Price'],
    np.nan
)
data['Stock_vs_XLP'] = np.where(
    data['XLP_Price'] != 0,
    data['PX_LAST'] / data['XLP_Price'],
    np.nan
)

# Momentum and volatility (with zero-division handling)
data['PX_ROC_9d'] = data['PX_LAST'].pct_change(9)
data['Revenue_ROC_20d'] = data['Revenue'].pct_change(20)

# Sentiment
data['ANR_Change_Abs'] = data['ANR Change'].abs()

# Price Momentum & Volatility
data['PX_LAST_MA21'] = data['PX_LAST'].rolling(21, min_periods=1).mean()
data['PX_LAST_MA63'] = data['PX_LAST'].rolling(63, min_periods=1).mean()
data['Price_Volatility_21d'] = data['PX_LAST'].rolling(21, min_periods=1).std()

# Fractal Efficiency (with zero-division handling)
diff_sum = data['PX_LAST'].rolling(21, min_periods=1).apply(lambda x: np.sum(np.abs(x.diff())))
data['Fractal_Efficiency_21d'] = np.where(
    diff_sum != 0,
    (data['PX_LAST'] - data['PX_LAST'].shift(21)) / diff_sum,
    np.nan
)

# Analyst Sentiment Dynamics
data['ANR_3d_change'] = data['ANR'].pct_change(3)
data['ANR_21d_zscore'] = (data['ANR'] - data['ANR'].rolling(21, min_periods=1).mean()) / data['ANR'].rolling(21, min_periods=1).std()
data['ANR_Target_Ratio'] = np.where(
    data['Target Price'] != 0,
    data['ANR'] / data['Target Price'],
    np.nan
)

# Seasonality Enhancements (with zero-division handling)
data['Q2_Premium'] = np.where(
    data['CQ2_Stock_Price'] != 0,
    data['PX_LAST'] / data['CQ2_Stock_Price'],
    np.nan
)
data['Q4_Discount'] = np.where(
    data['CQ4_Stock_Price'] != 0,
    data['PX_LAST'] / data['CQ4_Stock_Price'],
    np.nan
)
data['CQ2CQ4_Ratio_MA21'] = data['CQ2_CQ4_Seasonality_Ratio'].rolling(21, min_periods=1).mean()
data['CQ2PQ4_Ratio_ROC_14d'] = data['CQ2_PQ4_Seasonality_Ratio'].pct_change(14)

# Market-Relative Strength (with zero-division handling)
data['PBJ_RS_3d'] = np.where(
    data['Stock_vs_PBJ'].rolling(3, min_periods=1).mean() != 0,
    data['Stock_vs_PBJ'] / data['Stock_vs_PBJ'].rolling(3, min_periods=1).mean(),
    np.nan
)
data['XLP_RS_Volatility'] = data['Stock_vs_XLP'].rolling(21, min_periods=1).std()

# Risk Management Features
data['Max_Drawdown_21d'] = data['PX_LAST'].rolling(21, min_periods=1).apply(
    lambda x: (x.min() - x.max()) / x.max() if x.max() != 0 else np.nan
)
data['Recovery_Factor_63d'] = np.where(
    data['PX_LAST'].rolling(63, min_periods=1).mean() != 0,
    data['PX_LAST'].rolling(63, min_periods=1).max() / data['PX_LAST'].rolling(63, min_periods=1).mean(),
    np.nan
)

# Quarterly Interaction Features (with zero-division handling)
for q in ['Q_1', 'Q_2', 'Q_3', 'Q_4']:
    data[f'{q}_Price_Ratio'] = np.where(
        data[q] != 0,
        data['PX_LAST'] / data[q],
        np.nan
    )

# ================================================================================================
# FINAL CLEANUP
# ================================================================================================

# Replace inf and large values with NaN
data = data.replace([np.inf, -np.inf], np.nan)

# Forward fill NaNs (to preserve time-series continuity)
data.ffill(inplace=True)

# Drop any remaining NaNs
data.dropna(inplace=True)

# Display the engineered features
print("FEATURE ENGINEERING================================================================================================================================================")
print(data.head())
print(data.columns)
print(data.shape)
# print to csv
data.to_csv("output/feature_engineered_data.csv", index=False)









# Target variable
# ???subject to change???
# 10-day forward return
data['Future_20d_Return'] = data['PX_LAST'].shift(-20) / data['PX_LAST'] - 1
data['Label'] = data['Future_20d_Return'].apply(
    lambda x: 'Buy' if x > 0.04
                else 'Sell' if x < -0.04
                else 'Hold'
)
print("TARGET VARIABLE=====================================================================================================================================================")
print(data['Label'].value_counts())









# Final data preparation
data.ffill(inplace=True)  # Forward-fill missing values
data.dropna(inplace=True)  # Drop remaining NaN rows
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[['ANR', 'PX_LAST', 'Revenue', 'PBJ_Price']])

# Train test split to be time-series aware
train = data[data['Date'] < '2024-01-01']
test = data[data['Date'] >= '2024-01-01']

print("FINAL DATA PREPARATION=============================================================================================================================================")
print(train.head())









print("INITIALIZING FEATURES=====================================================================================================================================================")
# Feature selection
features = [
    'ANR', 'Target Price', 'ANR Change',
    'Revenue', 'Revenue_ROC', 'PX_LAST', 'HP_ROC', 'Stock Price', 'Q_1',
    'Q_2', 'Q_3', 'Q_4', 'CQ2_Stock_Price', 'CQ4_Stock_Price',
    'CQ2_CQ4_Seasonality_Ratio', 
    'Prev_Q4_Stock_Price', 'CQ2_PQ4_Seasonality_Ratio',
    'PBJ_Price', 'PBJ_ROC', 'PCUSEQTR_Price',
    'PCUSEQTR_ROC', 'VIX_Price', 'VIX_ROC', 'XLP_Price', 'XLP_ROC',
    'ANR_lag1', 'PX_LAST_lag1', 'Revenue_lag9', 'PX_LAST_MA9', 'ANR_MA20',
    'Target_Price_Gap', 'Undervalued', 'Stock_vs_PBJ', 'Stock_vs_XLP',
    'PX_ROC_9d', 'Revenue_ROC_20d', 'ANR_Change_Abs', 'PX_LAST_MA21',
    'PX_LAST_MA63', 'Price_Volatility_21d', 'Fractal_Efficiency_21d',
    'ANR_3d_change', 'ANR_21d_zscore', 'ANR_Target_Ratio', 'Q2_Premium',
    'Q4_Discount', 'CQ2CQ4_Ratio_MA21', 'CQ2PQ4_Ratio_ROC_14d', 'PBJ_RS_3d',
    'XLP_RS_Volatility', 'Max_Drawdown_21d', 'Recovery_Factor_63d',
    'Q_1_Price_Ratio', 'Q_2_Price_Ratio', 'Q_3_Price_Ratio',
    'Q_4_Price_Ratio'
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








# Multicollinearity removal process
print("\nREMOVING MULTICOLLINEAR FEATURES===================================================================================================================================")
features_modified = features.copy()
removal_occurred = True

while removal_occurred:
    # Calculate correlation matrix
    corr_matrix = X_train[features_modified].corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
    
    # Find high correlation pairs
    high_corr = [(col1, col2) for col1 in upper_triangle.columns 
                for col2 in upper_triangle.index 
                if upper_triangle.loc[col2, col1] > 0.6]
    
    if not high_corr:
        print("No highly correlated pairs remaining (r > 0.6).")
        removal_occurred = False
        break
    
    # Find highest correlation pair
    max_corr = 0
    max_pair = None
    for pair in high_corr:
        if upper_triangle.loc[pair[1], pair[0]] > max_corr:
            max_corr = upper_triangle.loc[pair[1], pair[0]]
            max_pair = (pair[0], pair[1])
    
    # Get feature importances
    rf_prelim = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_prelim.fit(X_train[features_modified], y_train_encoded)
    importances = pd.Series(rf_prelim.feature_importances_, index=features_modified)
    
    # Determine which feature to remove
    if importances[max_pair[0]] >= importances[max_pair[1]]:
        remove_feature = max_pair[1]
    else:
        remove_feature = max_pair[0]
    
    print(f"Removing '{remove_feature}' (importance: {importances[remove_feature]:.4f}) - Correlated with '{max_pair[0] if remove_feature == max_pair[1] else max_pair[1]}' (r = {max_corr:.2f})")
    
    # Update feature list
    features_modified.remove(remove_feature)
    
    # Update data splits
    X_train = train[features_modified]
    X_test = test[features_modified]

# Final feature set
features = features_modified
print("\nFINAL FEATURE SET:", features)










# ... [Previous code up through multicollinearity removal] ...

print("\nFINAL FEATURE SET AFTER MULTICOLLINEARITY REMOVAL:", features_modified)

# New: Remove low-importance features
print("\nREMOVING LOW-IMPORTANCE FEATURES===================================================================================================================================")

# Calculate feature importances with current set
rf_prelim = RandomForestClassifier(n_estimators=100, random_state=42)
rf_prelim.fit(X_train[features_modified], y_train_encoded)
importances = pd.Series(rf_prelim.feature_importances_, index=features_modified)

# Set dynamic thresholds (1% of max importance and absolute minimum)
max_importance = importances.max()
relative_threshold = max_importance * 0.01
absolute_threshold = 0.01  # Hard minimum regardless of max
low_importance = importances[
    (importances < relative_threshold) & 
    (importances < absolute_threshold)
].index.tolist()

# Remove low-importance features iteratively
while low_importance:
    # Remove the lowest importance feature first
    to_remove = importances.idxmin()
    print(f"Removing '{to_remove}' (importance: {importances[to_remove]:.4f})")
    features_modified.remove(to_remove)
    
    # Recalculate importances
    if len(features_modified) > 0:  # Prevent empty feature set
        rf_prelim.fit(X_train[features_modified], y_train_encoded)
        importances = pd.Series(rf_prelim.feature_importances_, index=features_modified)
        
        # Update low-importance list
        low_importance = importances[
            (importances < relative_threshold) & 
            (importances < absolute_threshold)
        ].index.tolist()
    else:
        break

# Final feature set update
features = features_modified
print("\nFINAL FEATURE SET AFTER IMPORTANCE FILTERING:", features)
if not features:
    raise ValueError("All features removed! Check threshold values.")

# Update data splits
X_train = train[features]
X_test = test[features]










print("TRAINING MODEL=====================================================================================================================================================")
# Initialize the model
rf = RandomForestClassifier(
    n_estimators=200,  # Number of trees
    max_depth=10,       # Control overfitting
    random_state=42,    # Reproducibility
    class_weight={0: 3, 1: 1, 2: 3}
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
# 1. Create a copy with only numeric features + encoded labels
data_corr = data[features].copy()
data_corr['Label'] = le.fit_transform(data['Label'])  # Encode labels numerically

# 2. Calculate correlations
corr_matrix = data_corr.corr()

# 3. Plot correlation matrix
plt.figure(figsize=(20, 16))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    annot_kws={"size": 8}
)
plt.title("Feature Correlation Matrix")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Get feature importances
importance = rf.feature_importances_
feature_names = X_train.columns

# Create a DataFrame for visualization
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values('Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.title('Random Forest Feature Importance')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis()  # Most important at top
plt.show()






# Use the model to make predictions