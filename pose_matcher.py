import cv2
import mediapipe as mp
import json
import numpy as np
import os

def load_reference_pose(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def normalize_landmarks(landmarks):
    # keeping ref point as midpoint between hips)
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
    score = max(0, 100 - (total_diff / count) * 1000)  # score out of 100
    return round(score, 2)

def get_pose_feedback(reference, live_normalized):
    feedback = []
   
    # Checking for arms straightness 
    left_elbow_ref = reference[13]
    left_elbow_live = live_normalized[13]
   
    vector_shoulder_to_elbow = np.array([left_elbow_ref['x'] - left_wrist_ref['x'],
                                        left_elbow_ref['y'] - left_wrist_ref['y']])
    vector_elbow_to_wrist = np.array([left_elbow_live['x'] - left_wrist_live['x'],
                                      left_elbow_live['y'] - left_wrist_live['y']])
    angle = np.arccos(np.dot(vector_shoulder_to_elbow, vector_elbow_to_wrist) /
                      (np.linalg.norm(vector_shoulder_to_elbow) * np.linalg.norm(vector_elbow_to_wrist)))
   
    if angle > np.pi / 3:  # keeping angle threshold as 60 degrees
        feedback.append("Relax your left elbow.")
   
    # add more checks for other body parts /*TODO*/
   
    return feedback


# pose loading
pose_name = input("Enter the name of the reference pose (e.g., araimandi): ")
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
        score = compare_poses(reference, live_normalized)
        cv2.putText(frame, f"Match Score: {score}%", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0) if score > 75 else (0, 0, 255), 3)
        
        #feedback
        feedback = get_pose_feedback(reference, live_normalized)
        if feedback:
            for i, message in enumerate(feedback):
                cv2.putText(frame, f"Feedback: {message}", (30, 120 + 30*i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        

    cv2.imshow("Live Pose Matching", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
