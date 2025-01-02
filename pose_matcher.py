import cv2
import mediapipe as mp
import json
import numpy as np
import os

def load_reference_pose(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def normalize_landmarks(landmarks):
    # Get reference point (midpoint between hips)
    left_hip = landmarks[23]
    right_hip = landmarks[24]

    center_x = (left_hip['x'] + right_hip['x']) / 2
    center_y = (left_hip['y'] + right_hip['y']) / 2

    scale = np.linalg.norm([
        landmarks[11]['x'] - landmarks[12]['x'],
        landmarks[11]['y'] - landmarks[12]['y']
    ]) + 1e-6

    for lm in landmarks:
        lm['x'] = (lm['x'] - center_x) / scale
        lm['y'] = (lm['y'] - center_y) / scale
    return landmarks


def compare_poses(ref, live):
    total_diff = 0
    count = 0
    for i in range(min(len(ref), len(live))):
        dx = ref[i]['x'] - live[i]['x']
        dy = ref[i]['y'] - live[i]['y']
        dist = np.sqrt(dx**2 + dy**2)
        total_diff += dist
        count += 1
    score = max(0, 100 - (total_diff / count) * 1000)  # Score out of 100
    return round(score, 2)

# Load your saved reference pose
pose_name = input("Enter the name of the reference pose (e.g., araimandi): ")
pose_path = f"reference_poses/{pose_name}.json"
if not os.path.exists(pose_path):
    print("Pose file not found.")
    exit()

reference = normalize_landmarks(load_reference_pose(pose_path))

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
drawing = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)
print("Get into your pose... Press ESC to exit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        live_landmarks = [
            {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}
            for lm in results.pose_landmarks.landmark
        ]

        live_normalized = normalize_landmarks(live_landmarks)
        score = compare_poses(reference, live_normalized)

        cv2.putText(frame, f"Match Score: {score}%", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0) if score > 75 else (0, 0, 255), 3)

    cv2.imshow("AI Natyam Coach - Pose Matching", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
