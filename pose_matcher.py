import cv2
import mediapipe as mp
import json
import numpy as np
import os
from scipy.spatial.distance import euclidean


def load_reference_pose(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def normalize_landmarks(landmarks):
    points = np.array([[lm['x'], lm['y']] for lm in landmarks])

    # using midpoint between hips as reference
    left_hip = points[23]
    right_hip = points[24]
    center = (left_hip + right_hip) / 2

    normalized = points - center

    # Scaling by distance between hips
    scale = np.linalg.norm(left_hip - right_hip)
    if scale < 1e-6:
        scale = 1e-6  # avoiding divide by zero

    normalized /= scale

    return normalized

def compare_poses(ref_norm, live_norm):
    if ref_norm.shape != live_norm.shape:
        return 0, ["Pose shape mismatch"]

    total_diff = 0
    feedback = []

    for i in range(len(ref_norm)):
        dist = euclidean(ref_norm[i], live_norm[i])
        total_diff += dist

    avg_diff = total_diff / len(ref_norm)
    score = max(0, 100 - avg_diff * 100)  # Tweak the multiplier as needed

    if score > 85:
        feedback.append("Perfect posture!\n")
    elif score > 70:
        feedback.append("Almost there! Few tweaks needed.\n")
    else:
        feedback.append("Keep practicing.\n")

    return round(score, 2), feedback



pose_name = input("Enter the name of the reference pose: ")
pose_path = f"reference_poses/{pose_name}.json"
if not os.path.exists(pose_path):
    print("Pose file not found.")
    exit()

reference = normalize_landmarks(load_reference_pose(pose_path))

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
drawing = mp.solutions.drawing_utils

# webcam code
cap = cv2.VideoCapture(0)
print("Strike a pose! Press ESC to exit")

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
        score, feedback = compare_poses(reference, live_normalized)

        cv2.putText(frame, f"Match Score: {score}%", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0) if score > 75 else (0, 0, 255), 3)

        feedback_text = "\n".join(feedback)
        cv2.putText(frame, feedback_text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Live Pose Matching", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()