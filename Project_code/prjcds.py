
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib



df = pd.read_csv("Project_code/smart_grid_dataframe.csv")

# Data Preprocessing and feature engineering

y = df["stabf"].map({"stable": 1, "unstable": 0})
X_raw = df.drop(columns=["stab", "stabf"], errors="ignore")

def features(data_df):
    df_feat = data_df.copy()
    for i in range(1, 5):
        df_feat[f'inertia_node{i}'] = df_feat[f'tau{i}'] * df_feat[f'p{i}']
    for i in range(1, 5):
        df_feat[f'elastic_volatility_node{i}'] = df_feat[f'g{i}'] * df_feat[f'p{i}']
    df_feat['system_latency_product'] = (
        df_feat['tau1'] * df_feat['tau2'] * df_feat['tau3'] * df_feat['tau4']
    )
    return df_feat



X_all = features(X_raw)

#  Train-Test Split BEFORE Scaling
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, stratify=y, random_state=42
)

#  Fit Scaler ONLY on Training Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  Model Training & Asset Saving
lr_model = LogisticRegression(class_weight='balanced', random_state=42)
lr_model.fit(X_train_scaled, y_train)

final_rf_model = RandomForestClassifier(
    n_estimators=250, 
    max_depth=20, 
    min_samples_split=5, 
    criterion='entropy', 
    class_weight='balanced', 
    random_state=42, 
    n_jobs=2)

final_rf_model.fit(X_train_scaled, y_train)

# Save lock files 
joblib.dump(lr_model, "lr_model.pkl")
joblib.dump(final_rf_model, "best_rf_model.pkl")
joblib.dump(scaler, 'scaler.pkl')

# XGBoost Model Training

final_xgb_model = xgb.XGBClassifier(
    n_estimators=350,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.75,
    colsample_bytree=0.8,
    scale_pos_weight=1.0,  
    random_state=42,
    n_jobs=2  # Prevents Streamlit thread-locking
)


final_xgb_model.fit(X_train_scaled, y_train)

# Evaluate on test set
y_pred_xgb = final_xgb_model.predict(X_test_scaled)

# Saving the model and scaler for later use in the Streamlit app
joblib.dump(final_xgb_model, 'best_xgb_model.pkl')  # Keeping the same filename or update to best_xgb_model.pkl
joblib.dump((X_test_scaled, y_test, X_all.columns.tolist()), 'Project_code/test_data.pkl')

print("Training complete! All model assets saved.")













