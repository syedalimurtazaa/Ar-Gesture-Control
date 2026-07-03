import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load Dataset
print("Loading dataset...")
df = pd.read_csv('gesture_data.csv')

# 2. Data Preprocessing (Lab 01)
# Checking for nulls (rare in this capture method, but good practice)
if df.isnull().values.any():
    print("Found missing values, dropping...")
    df = df.dropna()

X = df.drop('label', axis=1)
y = df['label']

# Split Data (80% Train, 20% Test) [cite: 143]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Random Forest Classifier (Lab 05)
print("Training Random Forest Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluation
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 5. Save Model
joblib.dump(model, 'ar_gesture_model.pkl')
print("Model saved as 'ar_gesture_model.pkl'")