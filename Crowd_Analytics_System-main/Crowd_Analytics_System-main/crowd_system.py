import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from collections import defaultdict
import pyttsx3
import winsound
import os

# 1. GENERATE THE AUDIO FILE ONCE 
audio_file = "overcrowded_alert.wav"
if not os.path.exists(audio_file):
    print("Generating alert audio file for the first time...")
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    # Save the spoken word directly to a .wav file
    engine.save_to_file("Overcrowded", audio_file)
    engine.runAndWait()

# 2. Alarm State Tracker
alarm_is_playing = False

# 3. Load YOLOv8 Nano
model = YOLO('yolov8n.pt')

# 4. Video Setup
video_path = "crowd.mp4" # Make sure this matches your video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video. Check the file path.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# --- CUSTOM ZONE SETUP ---
base_polygon = np.array([
    [int(width*0.3), int(height*0.4)], 
    [int(width*0.7), int(height*0.4)], 
    [int(width*0.8), int(height*0.8)], 
    [int(width*0.2), int(height*0.8)]  
], np.int32)

MOVE_X = 0  # Example: 150 moves it to the right. -150 moves it left.
MOVE_Y = 0  # Example: -50 moves it up. 50 moves it down.  
# Positive numbers move it Right/Down. Negative numbers move it Left/Up.

zone_polygon = base_polygon + np.array([MOVE_X, MOVE_Y])
zone_polygon = zone_polygon.reshape((-1, 1, 2))
# -------------------------

CROWD_THRESHOLD = 5
track_history = defaultdict(lambda: [])
density_data = []

# Output Window Configuration
cv2.namedWindow("Surveillance Feed", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Surveillance Feed", 960, 540) 

# Edge Map Sample
ret, sample_frame = cap.read()
if ret:
    gray = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    cv2.imwrite("scene_structure_edges.jpg", edges)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

out = cv2.VideoWriter('output_surveillance.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print("Starting AI Surveillance... Press 'q' on the video window to stop.")

# 5. Main Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break
        
    people_in_zone = 0
    results = model.track(frame, classes=[0], persist=True, verbose=False)
    
    cv2.polylines(frame, [zone_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
    overlay = frame.copy()
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            if cv2.pointPolygonTest(zone_polygon, (cx, cy), False) >= 0:
                people_in_zone += 1
                color = (0, 0, 255) 
            else:
                color = (0, 255, 0) 
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 4, color, -1)
            
            track = track_history[track_id]
            track.append((cx, cy))
            if len(track) > 20: 
                track.pop(0)
            cv2.polylines(frame, [np.array(track, np.int32).reshape((-1, 1, 2))], False, (0, 255, 255), 2)

    density_data.append(people_in_zone)
    
    # 6. PERFECTED ALERT LOGIC
    if people_in_zone > CROWD_THRESHOLD:
        cv2.fillPoly(overlay, [zone_polygon], (0, 0, 255))
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.putText(frame, "CROWD ALERT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        # If alarm isn't playing yet, start looping it
        if not alarm_is_playing:
            # SND_LOOP makes it repeat forever. SND_ASYNC keeps the video running.
            winsound.PlaySound(audio_file, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
            alarm_is_playing = True
            print(f"\n[ALERT] Overcrowded! {people_in_zone} people detected.")
            
    else:
        # If people drop below threshold and alarm is playing, STOP it instantly
        if alarm_is_playing:
            winsound.PlaySound(None, winsound.SND_PURGE) # Purge stops all active sounds
            alarm_is_playing = False
            print(f"\n[CLEAR] Crowd dispersed. {people_in_zone} people remaining.")

    cv2.putText(frame, f"Zone Count: {people_in_zone}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    print(f"Processing Frame... People in zone right now: {people_in_zone}   ", end='\r')
    
    out.write(frame)
    cv2.imshow("Surveillance Feed", frame) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources and stop any playing sounds before closing
winsound.PlaySound(None, winsound.SND_PURGE)
cap.release()
out.release()
cv2.destroyAllWindows()

# 7. Final Plot
if len(density_data) > 0:
    plt.figure(figsize=(10, 4))
    plt.plot(density_data, color='blue', label="Detected Persons")
    plt.axhline(y=CROWD_THRESHOLD, color='red', linestyle='--', label="Alert Threshold")
    plt.title("Zone Density Analysis")
    plt.xlabel("Frame Sequence")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.savefig("density_plot.png")
    print("\n\nExecution Completed! Check output_surveillance.mp4 and density_plot.png")