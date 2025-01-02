import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    print('Camera working!')
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        cv2.imshow('Webcam Test', frame)
        if cv2.waitKey(1) and 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()