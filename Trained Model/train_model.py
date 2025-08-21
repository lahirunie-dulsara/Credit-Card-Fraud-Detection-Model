import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load data
data = pd.read_csv('creditcard.csv')

# Scale Time and Amount
scaler_time = StandardScaler()
scaler_amount = StandardScaler()
data['scaled_time'] = scaler_time.fit_transform(data[['Time']])
data['scaled_amount'] = scaler_amount.fit_transform(data[['Amount']])
data = data.drop(['Time', 'Amount'], axis=1)

# Split features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Print feature names to verify
print("Training Feature Names:", X.columns.tolist())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate
y_pred = rf_model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model, scalers, and feature names
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(scaler_time, 'scaler_time.pkl')
joblib.dump(scaler_amount, 'scaler_amount.pkl')
joblib.dump(X.columns.tolist(), 'feature_names.pkl')
print("Model, scalers, and feature names saved!")