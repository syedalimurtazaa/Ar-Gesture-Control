import cv2
import mediapipe as mp
import csv
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# File setup
file_name = 'gesture_data.csv'

# Define headers (21 landmarks * 2 coordinates x,y + label)
headers = ['label']
for i in range(21):
    headers.extend([f'x{i}', f'y{i}'])

# Create CSV if not exists
if not os.path.exists(file_name):
    with open(file_name, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

cap = cv2.VideoCapture(0)
print("Press '0' for Open Hand, '1' for Fist, '2' for Pointing. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract normalized coordinates
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y])
            
            # Show on screen instruction
            cv2.putText(frame, "Press 0 (Open), 1 (Fist), 2 (Point)", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Capture key press for labeling
            key = cv2.waitKey(1)
            if key == ord('0'):
                with open(file_name, mode='a', newline='') as f:
                    csv.writer(f).writerow(['Open'] + coords)
                print("Captured: Open Hand")
            elif key == ord('1'):
                with open(file_name, mode='a', newline='') as f:
                    csv.writer(f).writerow(['Fist'] + coords)
                print("Captured: Fist")
            elif key == ord('2'):
                with open(file_name, mode='a', newline='') as f:
                    csv.writer(f).writerow(['Point'] + coords)
                print("Captured: Pointing")
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

    cv2.imshow("Data Collector", frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()