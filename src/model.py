import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import os
from collections import Counter

# Data Preprocessing ===========================================================================================================================================================================================
# Function to load data with error handling
def load_data(file_path, sheet_name):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        return pd.read_excel(file_path, sheet_name=sheet_name, skiprows=6)
    except Exception as e:
        raise ValueError(f"Error loading {file_path}: {e}")

# Load data
xlp = load_data("data/xlp.xlsx", "Worksheet")
pbj = load_data("data/pbj.xlsx", "Worksheet")
spx = load_data("data/spx.xlsx", "Worksheet")
mnst = load_data("data/mnst.xlsx", "Worksheet")
ko = load_data("data/ko.xlsx", "Worksheet")

# Clean and prepare data
def clean_data(df):
    df = df.dropna(subset=["Date"])
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df

xlp = clean_data(xlp)
pbj = clean_data(pbj)
spx = clean_data(spx)
mnst = clean_data(mnst)
ko = clean_data(ko)

# Merge data
data = pd.concat([xlp["PX_LAST"], pbj["PX_LAST"], spx["PX_LAST"], mnst["PX_LAST"], ko["PX_LAST"]], 
                 axis=1, keys=["XLP", "PBJ", "SPX", "MNST", "KO"])
data = data.dropna()
print("Data loaded and cleaned")
# Feature Engineering ===========================================================================================================================================================================================
# Calculate daily returns
returns = data.pct_change().dropna()

# Create moving averages
data["XLP_MA7"] = data["XLP"].rolling(window=7).mean()
data["PBJ_MA30"] = data["PBJ"].rolling(window=30).mean()

# Relative strength: Stock vs. XLP
data["MNST_XLP_RS"] = data["MNST"] / data["XLP"]
data["KO_XLP_RS"] = data["KO"] / data["XLP"]

# Drop rows with missing values
data = data.dropna()
print("Features engineered")
# Target Variable ===========================================================================================================================================================================================
# Define target for each stock
future_days = 5  # Look ahead 5 days
for stock in ["MNST", "KO"]:
    data[f"{stock}_Future_Price"] = data[stock].shift(-future_days)
    data[f"{stock}_Target"] = np.where(data[f"{stock}_Future_Price"] > data[stock] * 1.01, 1,  # Buy
                             np.where(data[f"{stock}_Future_Price"] < data[stock] * 0.99, -1,  # Sell
                             0))  # Hold

# Combine targets into a single column
data["Target"] = data[["MNST_Target", "KO_Target"]].mean(axis=1).round().astype(int)

# Drop rows with missing targets
data = data.dropna()
print("Target variable defined")

# Check the distribution of the target variable
print("Target distribution:\n", data["Target"].value_counts())
# Model Training ===========================================================================================================================================================================================
# Features: Use PBJ, XLP, SPX, and stock-specific features
print("Training and testing data...")
features = ["PBJ", "XLP", "SPX", "XLP_MA7", "PBJ_MA30", "MNST_XLP_RS", "KO_XLP_RS"]

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(data[features])
y = data["Target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
rf = RandomForestClassifier(random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate
y_pred = best_model.predict(X_test)
print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Check feature importance
feature_importances = pd.Series(best_model.feature_importances_, index=features)
print("Feature Importances:\n", feature_importances)

# Save model
model = best_model
print("Model trained and tested")
# Predictions ===========================================================================================================================================================================================
# Predicting
def predict_stock_signal(new_stock_data, model, indicators, scaler):
    # Merge new stock data with indicators
    data = pd.concat([new_stock_data, indicators], axis=1)
    data = data.dropna()
    
    # Feature engineering
    data["XLP_MA7"] = data["XLP"].rolling(window=7).mean()
    data["PBJ_MA30"] = data["PBJ"].rolling(window=30).mean()
    data["MNST_XLP_RS"] = data["Stock"] / data["XLP"]
    data["KO_XLP_RS"] = data["Stock"] / data["XLP"]
    
    # Normalize features
    X = scaler.transform(data[["PBJ", "XLP", "SPX", "XLP_MA7", "PBJ_MA30", "MNST_XLP_RS", "KO_XLP_RS"]])
    predictions = model.predict(X)
    
    return predictions

# Function to count majority prediction and print recommendation
def get_majority_recommendation(predictions):
    # Count the occurrences of each prediction
    prediction_counts = Counter(predictions)
    
    # Get the majority prediction
    majority_prediction = prediction_counts.most_common(1)[0][0]  # Most frequent prediction
    
    # Map the majority prediction to a recommendation
    if majority_prediction == 1:
        return "Buy"
    elif majority_prediction == 0:
        return "Hold"
    elif majority_prediction == -1:
        return "Sell"
    else:
        return "No clear recommendation"

# Predict for new stock data
input_folder = "input"
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

for file_name in os.listdir(input_folder):
    if file_name.endswith(".xlsx"):
        file_path = os.path.join(input_folder, file_name)
        input_stock = pd.read_excel(file_path, sheet_name="Worksheet", skiprows=6)  # Load new stock data
        input_stock = clean_data(input_stock)
        input_stock.rename(columns={"PX_LAST": "Stock"}, inplace=True)

        # Predict
        predictions = predict_stock_signal(input_stock, model, data[["PBJ", "XLP", "SPX"]], scaler)
        print(f"Predictions for {file_name}:", predictions)
        recommendation = get_majority_recommendation(predictions)
        print(f"Recommendation for {file_name}: {recommendation}")

        # Save predictions to CSV
        output_file_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_predictions.csv")
        pd.DataFrame(predictions, columns=["Prediction"]).to_csv(output_file_path, index=False)
        print(f"Predictions saved to {output_file_path}")