import cv2
import mediapipe as mp
import joblib
import numpy as np

# Load the trained model
model = joblib.load('ar_gesture_model.pkl')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# AR Object Properties (Initial State)
ar_object_color = (255, 0, 0) # Blue
ar_text = "Status: Idle"
zoom_level = 1.0

cap = cv2.VideoCapture(0)

print("Starting AR Interface...")
while True:
    ret, frame = cap.read()
    if not ret: break
    
    # 1. Preprocessing (Flip & Color)
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    prediction = "None"
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract features for prediction
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y])
            
            # 2. Prediction (Lab 05 & 13)
            # Reshape for single sample prediction
            prediction = model.predict([coords])[0]
            
            # 3. AR Control Logic (Lab 14 Integration)
            if prediction == "Open":
                ar_text = "AR Action: Reset/Idle"
                ar_object_color = (255, 0, 0) # Blue
                zoom_level = 1.0
                
            elif prediction == "Fist":
                ar_text = "AR Action: Activate/Hold"
                ar_object_color = (0, 0, 255) # Red
                
            elif prediction == "Point":
                ar_text = "AR Action: Zoom In"
                ar_object_color = (0, 255, 0) # Green
                zoom_level = min(zoom_level + 0.02, 2.0) # Simulate Zoom

    # 4. Render AR Overlay (Lab 12)
    # Draw a virtual "box" that reacts to gestures
    h, w, c = frame.shape
    center_x, center_y = int(w/2), int(h/2)
    box_size = int(100 * zoom_level)
    
    cv2.rectangle(frame, 
                  (center_x - box_size, center_y - box_size), 
                  (center_x + box_size, center_y + box_size), 
                  ar_object_color, -1)
    
    # Overlay Info
    cv2.putText(frame, f"Gesture: {prediction}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, ar_text, (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("AR Gesture Control - Lab 14", frame)
    
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()