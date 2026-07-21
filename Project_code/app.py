import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

st.set_page_config(page_title="Smart Power Grid", layout="wide")
st.title('⚡ Smart Power Grid Solution')

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

# Load Assets via Cache

@st.cache_resource
def load_all_assets():
    # Dynamically locate the folder where app.py lives (Project_code/)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
    lr = joblib.load(os.path.join(BASE_DIR, 'lr_model.pkl'))
    rf = joblib.load(os.path.join(BASE_DIR, 'best_rf_model.pkl'))
    xgb_m = joblib.load(os.path.join(BASE_DIR, 'best_xgb_model.pkl'))
    X_test_s, y_test_s, feature_names = joblib.load(os.path.join(BASE_DIR, 'test_data.pkl'))
    
    return scaler, lr, rf, xgb_m, X_test_s, y_test_s, feature_names 


scaler, lr_model, final_rf_model, final_xgb_model, X_test_scaled, y_test, feature_cols = load_all_assets()

# STREAMLIT DASHBOARD UI 
# Sidebar Interactive Controls
st.sidebar.header("🎛️ Input Smart Grid Parameters")

st.sidebar.subheader("⏱️ Reaction Times (tau)")
tau1 = st.sidebar.slider("tau1 (Node 1 -- Source)", 0.5, 10.0, 5.0)
tau2 = st.sidebar.slider("tau2 (Node 2 -- Consumer)", 0.5, 10.0, 5.0)
tau3 = st.sidebar.slider("tau3 (Node 3 -- Consumer)", 0.5, 10.0, 5.0)
tau4 = st.sidebar.slider("tau4 (Node 4 -- Consumer)", 0.5, 10.0, 5.0)

st.sidebar.subheader("🔌 Nominal Power (p)")
p2 = st.sidebar.slider("p2 (Consumer 1)", -2.0, -0.5, -1.0)
p3 = st.sidebar.slider("p3 (Consumer 2)", -2.0, -0.5, -1.0)
p4 = st.sidebar.slider("p4 (Consumer 3)", -2.0, -0.5, -1.0)
p1 = -(p2 + p3 + p4)  # Ensure grid balance constraint #Produced total power 
st.sidebar.caption(f"Calculated p1 (Producer): `{p1:.2f}`")

st.sidebar.subheader("📈 Price Elasticity (g)")
g1 = st.sidebar.slider("g1 (Node 1)", 0.05, 1.0, 0.5)
g2 = st.sidebar.slider("g2 (Node 2)", 0.05, 1.0, 0.5)
g3 = st.sidebar.slider("g3 (Node 3)", 0.05, 1.0, 0.5)
g4 = st.sidebar.slider("g4 (Node 4)", 0.05, 1.0, 0.5)

# Real-Time Inference on User Input
input_base_df = pd.DataFrame([{
    'tau1': tau1, 'tau2': tau2, 'tau3': tau3, 'tau4': tau4,
    'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4,
    'g1': g1, 'g2': g2, 'g3': g3, 'g4': g4
}])

# Compute engineered features for the input
input_full_df = features(input_base_df)

# Scale and Predict
input_scaled = scaler.transform(input_full_df)
prediction = final_xgb_model.predict(input_scaled)[0]
probabilities = final_xgb_model.predict_proba(input_scaled)[0]
confidence = np.max(probabilities) * 100

# Output Predictions
st.subheader("🔮 Stability Assessment")
st.write("Based on XGBoost Model Prediction:")
col1, col2 = st.columns(2)

with col1:
    if prediction == 1:
        st.success("### Status: STABLE ✔️🟢")
    else:
        st.error("### Status: UNSTABLE❕🔴")

with col2:
    st.metric(label="Model Confidence", value=f"{confidence:.2f}%")
    st.progress(int(confidence))


# Check exact model type loaded into memory
st.sidebar.info(f"Active Model: `{type(final_xgb_model).__name__}`")


# Classification Reports for All Models

st.subheader("📊 Baseline Performance")
st.text("Logistic Regression Classification Report:")
st.dataframe(classification_report(y_test, lr_model.predict(X_test_scaled), output_dict=True))

st.subheader("🌲Random Forest Classification Performance")
st.text("Random Forest Classification Report:")
st.dataframe(classification_report(y_test, final_rf_model.predict(X_test_scaled), output_dict=True))

st.subheader("🔥XGBoost Classification Performance")
st.text("XGBoost Classification Report:")
st.dataframe(classification_report(y_test, final_xgb_model.predict(X_test_scaled), output_dict=True))


#  PREPARING EVALUATION METRICS 
models = {
    'Logistic Regression': (lr_model, X_test_scaled),
    'Random Forest': (final_rf_model, X_test_scaled),
    'XGBoost': (final_xgb_model, X_test_scaled)
}

results = []
for name, (mdl, X_data) in models.items():
    preds = mdl.predict(X_data)
    probs = mdl.predict_proba(X_data)[:, 1]
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, preds),
        'ROC-AUC': roc_auc_score(y_test, probs)
    })

metrics_df = pd.DataFrame(results)

# PLOT 1: MODEL PERFORMANCE COMPARISON BAR CHART 
st.subheader("📈 Overall Model Performance Comparison")

fig_bar = px.bar(
    metrics_df.melt(id_vars=['Model'], value_vars=['Accuracy', 'ROC-AUC']),
    x='Model',
    y='value',
    color='variable',
    barmode='group',
    text_auto='.3f',
    title="Accuracy and ROC-AUC Scores Across Models",
    color_discrete_sequence=['#636EFA', '#00CC96']
)
fig_bar.update_layout(yaxis_range=[0.8, 1.0], yaxis_title="Score")
st.plotly_chart(fig_bar, use_container_width=True)



# PLOT 3: TOP 10 FEATURE IMPORTANCES (RF VS XGBOOST) 
st.subheader("🔥 Key Driver Comparison (Feature Importances)")

rf_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': final_rf_model.feature_importances_, 'Model': 'Random Forest'})
xgb_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': final_xgb_model.feature_importances_, 'Model': 'XGBoost'})

imp_df = pd.concat([rf_imp, xgb_imp])
top_features = rf_imp.nlargest(10, 'Importance')['Feature']
imp_df = imp_df[imp_df['Feature'].isin(top_features)]

fig_imp = px.bar(
    imp_df,
    x='Importance',
    y='Feature',
    color='Model',
    barmode='group',
    orientation='h',
    title="Top 10 Influential Features (Random Forest vs. XGBoost)"
)
fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig_imp, use_container_width=True)