# Real-Time AR Control via Gesture Recognition 🖐️🤖

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-green)
![Computer Vision](https://img.shields.io/badge/CV-MediaPipe-orange)

This project implements a Machine Learning pipeline to control an Augmented Reality (AR) interface using real-time hand gestures. It captures hand landmarks via webcam, classifies them using a Random Forest model, and triggers dynamic AR visual effects based on the recognized gesture.

## 🚀 Features

* **Real-Time Detection:** High-speed hand tracking using MediaPipe (21 3D landmarks).
* **Custom Dataset:** Includes tools to record and label your own gesture data.
* **ML Classification:** Random Forest model achieving ~95% accuracy.
* **AR Visuals:** Dynamic overlay that responds to gestures:
    * 🖐️ **Open Hand:** Reset / Idle (Blue Box)
    * ✊ **Fist:** Hold / Activate (Red Box)
    * ☝️ **Pointing:** Zoom In (Green Box)

## 🛠️ Tech Stack

* **Language:** Python
* **Computer Vision:** OpenCV, MediaPipe
* **Machine Learning:** Scikit-Learn
* **Data Handling:** Pandas, NumPy
* **Model Persistence:** Joblib

## 📂 Project Structure

```bash
├── data_collector.py    # Script to capture hand landmarks and save to CSV
├── gesture_data.csv     # The dataset (coordinates of hand landmarks)
├── train_model.py       # Script to train the Random Forest Classifier
├── ar_gesture_model.pkl # The trained ML model file
├── main_ar_app.py       # Main application running the AR interface
```

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/sarimraza890/AR_Gesture_Control.git](https://github.com/sarimraza890/AR_Gesture_Control.git)
    cd AR_Gesture_Control
    ```

2.  **Set up a Virtual Environment (Recommended):**
    ```bash
    # Windows
    python -m venv hand_recognition
    .\hand_recognition\Scripts\activate

    # Mac/Linux
    python3 -m venv hand_recognition
    source hand_recognition/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install opencv-python mediapipe scikit-learn pandas numpy joblib matplotlib seaborn
    ```
    *(Note: If you encounter MediaPipe errors on Windows, try: `pip install "mediapipe==0.10.9" "protobuf==3.20.3"`)

## 🏃‍♂️ Usage Guide

**Step 1: Data Collection**
Run the collector to create your dataset.
```bash
python data_collector.py
```
* Press **'0'** to record **Open Hand**.
* Press **'1'** to record **Fist**.
* Press **'2'** to record **Pointing**.
* Press **'q'** to quit.

**Step 2: Model Training**
Train the Random Forest model on your new data.
```bash
python train_model.py
```
* This generates the `ar_gesture_model.pkl` file and prints the accuracy score.

**Step 3: Run AR Interface**
Launch the real-time recognition app.
```bash
python main_ar_app.py
```
* Show your hand to the camera to see the AR overlay react!

## 📊 Results

The model typically achieves **95%+ accuracy** with a balanced dataset.

* **Confusion Matrix:** Minimal misclassification between distinct poses.
* **Feature Importance:** The model relies heavily on the **Index Finger Tip** and **Thumb Tip** coordinates to distinguish between the three gestures.

---
