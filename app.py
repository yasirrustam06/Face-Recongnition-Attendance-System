import tkinter as tk
import cv2
import numpy as np
import threading
import tensorflow as tf
from keras.models import load_model
import csv
from datetime import datetime

class_names = ['Wasem', 'lionel_messi', 'maria_sharapova', 'roger_federer', 'virat_kohli']

model = load_model("model/faceRecong.h5")

root = tk.Tk()
root.geometry('1000x1000')
root.title("Face Recognition APP")

main_frame = tk.Frame(root, highlightbackground='green', highlightthickness=5)
main_frame.pack(side=tk.LEFT)
main_frame.pack_propagate(False)
main_frame.configure(height=1000, width=1000)

home_frame = tk.Frame(main_frame)
lb = tk.Label(home_frame, text="Press the Runweb_cam to run the app", font=("Bold", 20))
lb.pack()
webcam_on = False

def show_webcam():
    global webcam_on
    webcam_on = True
    button_show.config(bg='orange')
    button_hide.config(bg='white')
    t = threading.Thread(target=run_webcam)
    t.start()

def hide_webcam():
    global webcam_on
    webcam_on = False
    button_show.config(bg='white')
    button_hide.config(bg='orange')

def stop_app():
    root.destroy()

def save_attendance(name):
    now = datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    with open('attendance.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, time])

def run_webcam():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    attendance_recorded = False
    while webcam_on:
        ret, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Load the pre-trained face detection classifier from OpenCV
        face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        img_res = cv2.resize(img, (100, 100))
        norm_img = img_res / 255.0
        img_exp = np.expand_dims(norm_img, 0)
        result = model.predict(img_exp)
        pred = np.argmax(result)
        label = class_names[pred]
        cv2.putText(img, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
        cv2.imshow("Image", img)

        if pred != 0 and not attendance_recorded:  # Check if a recognized person's index is not 0 (background class)
            save_attendance(label)
            attendance_recorded = True
        elif pred == 0:
            attendance_recorded = False

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    button_show.config(bg='white')

button_show = tk.Button(home_frame, text="Runweb_cam", command=show_webcam, bg='orange')
button_show.pack(pady=20)

button_hide = tk.Button(home_frame, text="Hide Webcam", command=hide_webcam, bg='orange')
button_hide.pack(pady=20)

button_stop = tk.Button(home_frame, text="Stop App", command=stop_app, bg='orange')
button_stop.pack(pady=20)

home_frame.pack(pady=20)

root.mainloop()
