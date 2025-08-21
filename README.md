# Credit-Card-Fraud-Detection-Model

# 🛡️ Credit Card Fraud Detection

This project implements a machine learning solution to identify fraudulent credit card transactions using a **Random Forest Classifier**.  
It is built with **Python, scikit-learn, pandas**, and features a **Streamlit** interface for real-time predictions.  

The model achieves **73% recall** and **96% precision** for fraud detection on the Kaggle Credit Card Fraud Detection dataset.

---

## 📊 Dataset

**Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)  

- ~284,807 anonymized transactions  
- Features:  
  - `Time`: seconds since first transaction  
  - `Amount`: transaction value  
  - `V1–V28`: PCA-transformed features  
  - `Class`: 0 = non-fraud, 1 = fraud (~0.17% of cases)  

⚠️ The dataset is **highly imbalanced**, making fraud detection challenging.

---

## ⚙️ Model Training

Training pipeline:

1. **Preprocessing**
   - Scale `Time` and `Amount` using `StandardScaler`
   - Retain `V1–V28`, drop original columns  

2. **Class Imbalance**
   - Apply `class_weight="balanced"` in Random Forest  

3. **Training**
   - 80/20 train-test split with stratified sampling  
   - Random Forest with 100 estimators  
   - Achieved **73% recall**, **96% precision**

4. **Output**
   - Saves:
     - `rf_model.pkl` – trained model  
     - `scaler_time.pkl`, `scaler_amount.pkl` – scalers  
     - `feature_names.pkl` – feature list  

---

## 📂 Key Files

- `train_model.py` – preprocess, train, and save model  
- `creditcard.csv` – dataset (download separately from Kaggle)  
- `app.py` – Streamlit UI  
- `ui_screenshot.png` – UI screenshot (see below)  

---

## 🎨 User Interface

Built with **Streamlit**, the UI provides:

1. **Manual Input**
   - Enter **Amount (USD)** and **Time (hours since midnight)**
   - Uses mean `V1–V28` values to predict fraud  
   - Displays result as:
     - ✅ **Non-Fraudulent** (green)  
     - ❌ **Fraudulent** (red)  
   - Shows fraud probability  

2. **CSV Upload**
   - Upload file with `Time`, `V1–V28`, `Amount`  
   - Validates format, scales inputs, predicts fraud  
   - Recommended for accurate results  

### Screenshot
![UI Screenshot](User_Interface/ui_screenshot.png)

---

## 📌 Notes

- **Performance**  
  - Recall = **73%** (detects most fraud cases)  
  - Precision = **96%** (few false positives)  
- Manual input may bias toward **non-fraud** due to mean `V1–V28` values.  
- **CSV upload** is recommended for accurate results.  

---

## 🔮 Improvements

- Apply **SMOTE** to improve recall  
- Add **visualizations** in UI (confusion matrix, ROC curve)  
- Deploy to **cloud platforms** (Heroku, ngrok, AWS)  

