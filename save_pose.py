import cv2
import mediapipe as mp
import json
import os

pose_name = input("Enter a name for the pose (e.g., araimandi): ")

# Create output folder if not exists
output_dir = "reference_poses"
os.makedirs(output_dir, exist_ok=True)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

landmarks_data = []

print("Press 's' to save the current pose when ready. Press ESC to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Save Pose - AI Natyam Coach', frame)

    key = cv2.waitKey(5)

    if key == ord('s') and results.pose_landmarks:
        # Save landmarks
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append({
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
                'visibility': lm.visibility
            })
        with open(f"{output_dir}/{pose_name}.json", "w") as f:
            json.dump(landmarks, f, indent=2)
        print(f"Pose saved as {pose_name}.json")
        break

    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()