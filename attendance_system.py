import cv2
import dlib
import numpy as np
import os
import pandas as pd
import threading
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
import speech_recognition as sr

# Initialize face detector and recognition model
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model.dat')

class AttendanceApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Attendance System")
        self.stop_flag = False
        self.session_logged_names = set()  # Track logged names for the current session

        # UI Buttons
        self.start_button = tk.Button(master, text="Start Attendance", command=self.start_attendance)
        self.start_button.pack()

        self.stop_button = tk.Button(master, text="Stop Attendance", command=self.stop_attendance)
        self.stop_button.pack()

        self.log_button = tk.Button(master, text="View Attendance Log", command=self.view_log)
        self.log_button.pack()

        self.speech_button = tk.Button(master, text="Speech Roll Call", command=self.speech_roll_call)
        self.speech_button.pack()

    def train_model(self, data_folder):
        known_face_encodings = []
        known_face_names = []
        for student_folder in os.listdir(data_folder):
            student_path = os.path.join(data_folder, student_folder)
            for image_name in os.listdir(student_path):
                img = dlib.load_rgb_image(os.path.join(student_path, image_name))
                dets = detector(img, 1)
                for d in dets:
                    shape = sp(img, d)
                    face_encoding = facerec.compute_face_descriptor(img, shape)
                    known_face_encodings.append(np.array(face_encoding))
                    known_face_names.append(student_folder)
        return known_face_encodings, known_face_names

    def log_attendance(self, face_names):
        if not os.path.exists('attendance_log.csv'):
            with open('attendance_log.csv', 'w') as f:
                f.write("Name,Timestamp\n")
        with open('attendance_log.csv', 'a') as f:
            for name in face_names:
                if name != "Unknown" and name not in self.session_logged_names:
                    f.write(f"{name},{datetime.now()}\n")
                    self.session_logged_names.add(name)  # Mark name as logged for the session

    def capture_attendance(self, known_face_encodings, known_face_names):
        self.stop_flag = False
        self.session_logged_names.clear()  # Clear logged names for a new session
        video_capture = cv2.VideoCapture(0)

        while not self.stop_flag:
            ret, frame = video_capture.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = detector(rgb_frame, 1)
            face_encodings = [np.array(facerec.compute_face_descriptor(rgb_frame, sp(rgb_frame, d))) for d in face_locations]
            
            face_names = []
            for face_encoding in face_encodings:
                distances = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
                min_distance = min(distances) if len(distances) > 0 else float('inf')
                name = "Unknown"
                if min_distance < 0.6:
                    best_match_index = np.argmin(distances)
                    name = known_face_names[best_match_index]
                face_names.append(name)

            self.log_attendance(face_names)

            for (d, name) in zip(face_locations, face_names):
                left, top, right, bottom = d.left(), d.top(), d.right(), d.bottom()
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, bottom - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Attendance System', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

    def start_attendance(self):
        known_face_encodings, known_face_names = self.train_model('attendance_data')
        threading.Thread(target=self.capture_attendance, args=(known_face_encodings, known_face_names), daemon=True).start()

    def stop_attendance(self):
        self.stop_flag = True
        messagebox.showinfo("Info", "Attendance capture stopped.")

    def view_log(self):
        if not os.path.exists('attendance_log.csv'):
            messagebox.showinfo("Attendance Log", "No attendance records found.")
            return
        with open('attendance_log.csv', 'r') as f:
            log_data = f.read()
        messagebox.showinfo("Attendance Log", log_data)

    def speech_roll_call(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            messagebox.showinfo("Speech Recognition", "Listening...")
            audio = recognizer.listen(source)
        try:
            name = recognizer.recognize_google(audio)
            if name not in self.session_logged_names:
                self.log_attendance([name])
                messagebox.showinfo("Speech Recognition", f"Attendance marked for {name}")
            else:
                messagebox.showinfo("Speech Recognition", f"{name} is already marked present.")
        except sr.UnknownValueError:
            messagebox.showerror("Speech Recognition", "Could not understand audio.")
        except sr.RequestError:
            messagebox.showerror("Speech Recognition", "Could not request results.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()
