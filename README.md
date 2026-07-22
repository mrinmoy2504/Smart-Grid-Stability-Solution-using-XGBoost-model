# ⚡ Smart Grid Stability Prediction using XGBoost

An interactive Machine Learning web application designed to predict the stability of a 4-node star-shaped electrical power grid based on system parameters. 

This project explores the intersection of Electrical Engineering principles and modern Data Science by comparative analysis of multiple classification algorithms, featuring an **XGBoost Classifier** core deployed live via Streamlit.

---

## 🔗 Live Application
Access the deployed interactive web dashboard here:  
👉 **[Smart Grid Stability App](https://smart-grid-stability-solution-using-xgboost-model-g3ymygul6gv5.streamlit.app/)**

---

## 📌 Features & Workflow

* **Dataset Preprocessing**: Structured feature scaling and clean preprocessing of grid parameters (e.g., reaction times, power generation/consumption balance, price elasticity).
* **Model Benchmarking**: Comprehensive comparative evaluation of three ML models:
  * **Logistic Regression** (Linear baseline)
  * **Random Forest Classifier** (Bagging ensemble)
  * **XGBoost Classifier** (eXtreme Gradient Boosting - Selected for best accuracy/AUC)
* **Performance Evaluation**: Features ROC-AUC curve visualizations and evaluation metrics comparing model performances.
* **Modular Codebase**: Pipeline functions separated cleanly inside the `Project_code/` directory for maintainability.
* **Interactive UI**: Built with Streamlit to enable real-time user input tuning and instant grid stability classification.

---
