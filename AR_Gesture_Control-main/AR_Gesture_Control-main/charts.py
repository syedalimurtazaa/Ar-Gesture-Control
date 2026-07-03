import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import joblib

# Set visual style
sns.set_style("whitegrid")

# 1. Load your ACTUAL data
try:
    df = pd.read_csv('gesture_data.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'gesture_data.csv' not found. Please run data_collector.py first.")
    exit()

# --- CHART 1: Class Distribution ---
plt.figure(figsize=(8, 6))
class_counts = df['label'].value_counts()
sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
plt.title('Dataset Class Distribution', fontsize=14)
plt.xlabel('Gesture Class', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.savefig('class_distribution.png')
print("Saved: class_distribution.png")
plt.close()

# --- Prepare Data for Model Charts ---
X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load or Train Model
# We train a fresh one here just to ensure we have the object ready for plotting
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- CHART 2: Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
labels = sorted(y.unique())

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix (Test Set)', fontsize=14)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.savefig('confusion_matrix.png')
print("Saved: confusion_matrix.png")
plt.close()

# --- CHART 3: Feature Importance ---
# We have 42 features. We will plot the top 10.
importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1][:10] # Top 10

plt.figure(figsize=(10, 6))
plt.barh(range(10), importances[indices], align='center', color='purple')
plt.yticks(range(10), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance', fontsize=12)
plt.title('Top 10 Feature Importances', fontsize=14)
plt.gca().invert_yaxis() # Highest at top
plt.savefig('feature_importance.png')
print("Saved: feature_importance.png")
plt.close()

print("\nAll charts generated successfully!")