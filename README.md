# Smart Attendance System Using Machine Learning
This project is a **Machine Learning-based Smart Attendance System** that automates professor attendance tracking using **Face Recognition** and **Speech Roll Call**. The system utilizes **dlib's face recognition model** and **speech recognition** to accurately mark attendance.

## 🚀 Features

✔️ **Face Recognition** – Automatically detects and recognizes faces.  
✔️ **Speech-Based Roll Call** – Mark attendance via voice input.  
✔️ **Live Attendance Logging** – Captures attendance in real-time.  
✔️ **CSV Attendance Logs** – Stores attendance in `attendance_log.csv`.  
✔️ **User-Friendly GUI** – Simple interface for easy use.  

## 🛠 Installation

### 📌 Prerequisites
Ensure you have Python 3.x installed, along with the following libraries:
pip install opencv-python dlib numpy pandas speechrecognition
📥 Download Pretrained Models
dlib_face_recognition_resnet_model.dat (Face recognition model)
shape_predictor_68_face_landmarks.dat (Facial landmark detector)
Place these files in your project directory.
📌 Usage
1️⃣ Collect Training Data
Run attendance_data.py to capture facial images for training.
python attendance_data.py
🔹 Modify student_name before running to store images in a unique folder.
2️⃣ Start the Attendance System
Run attendance_system.py to launch the face recognition-based attendance system.
python attendance_system.py
3️⃣ View Attendance Logs
Attendance records are saved in attendance_log.csv with timestamps.
📂 Project Structure
📁 Smart-Attendance-System
│── attendance_data.py          # Captures training images
│── attendance_system.py        # Face recognition & speech attendance system
│── attendance_log.csv          # Stores attendance records
│── dlib_face_recognition_resnet_model.dat   # Pretrained face recognition model
│── shape_predictor_68_face_landmarks.dat    # Face landmark detector
🚀 Future Enhancements
🔹 Deep Learning Model – Improve accuracy with CNN-based models.
🔹 Mobile Integration – Develop an Android/iOS app for remote access.
🔹 Cloud Support – Store attendance records in a database.

📝 License
This project is licensed under the MIT License.

👨‍💻 Contributors
[SAHIL KUMAR]







