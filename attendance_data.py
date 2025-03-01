import cv2
import os

# Set student name
student_name = "rishu modi "  # Change name for each student

# Create directory if not exists
data_path = f'attendance_data/{student_name}'
os.makedirs(data_path, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)

count = 0
while count < 5:  # Capture 5 images
    ret, frame = cap.read()
    if not ret:
        break

    img_path = os.path.join(data_path, f'image{count + 1}.jpg')
    cv2.imwrite(img_path, frame)
    count += 1

    cv2.imshow('Capturing Images', frame)
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Images saved in {data_path}")
