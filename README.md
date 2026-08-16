# ✋ Hand Gesture Recognition System

A real-time **Hand Gesture Recognition System** developed using **Python, OpenCV, Computer Vision, and Machine Learning** to detect and recognize different hand gestures through camera input.

The system analyzes hand positions and identifies gestures such as **V**, **Index Pointing**, and **L Right** in real time.

---

## 📌 Project Overview

Hand Gesture Recognition is a Computer Vision project designed to recognize human hand gestures from live camera input.

The application captures hand movements through a webcam, detects the hand region, extracts gesture-related features, and predicts the corresponding gesture.

This project demonstrates the practical use of **Computer Vision and Machine Learning for Human-Computer Interaction (HCI)**.

---

## 🎯 Objective

The main objectives of this project are:

* Detect human hands using real-time camera input
* Identify different hand gestures
* Process live video frames
* Extract important hand features
* Classify hand gestures accurately
* Display the detected gesture in real time
* Demonstrate touchless human-computer interaction

---

## 🚀 Features

* Real-time hand detection
* Live webcam processing
* Gesture recognition
* Hand region detection
* Image preprocessing
* Feature extraction
* Gesture classification
* Real-time prediction display
* Simple and easy-to-understand output

---

## 🛠️ Technologies Used

* **Python**
* **OpenCV**
* **Computer Vision**
* **Machine Learning**
* **NumPy**
* **Image Processing**
* **VS Code / Jupyter Notebook**

---

## ⚙️ How the System Works

The Hand Gesture Recognition System follows the workflow below:

```text
Webcam Input
     ↓
Capture Video Frame
     ↓
Image Preprocessing
     ↓
Hand Detection
     ↓
Feature Extraction
     ↓
Gesture Classification
     ↓
Predicted Gesture
     ↓
Display Result
```

### Step 1: Webcam Input

The system captures live video using the computer webcam.

### Step 2: Image Processing

Each frame captured from the webcam is processed using Computer Vision techniques.

### Step 3: Hand Detection

The system identifies the hand region from the video frame.

### Step 4: Feature Extraction

Important features related to the hand position and finger arrangement are extracted.

### Step 5: Gesture Classification

The extracted features are analyzed to determine the corresponding hand gesture.

### Step 6: Display Prediction

The predicted gesture is displayed on the video output in real time.

---

## ✋ Recognized Gestures

The current system demonstrates recognition of gestures including:

| Gesture             | Prediction       |
| ------------------- | ---------------- |
| ✌️ Two Fingers      | `V`              |
| ☝️ Index Finger     | `Index_Pointing` |
| 👆 L-Shaped Gesture | `L_right`        |

---

## 📸 Project Output

The following output demonstrates real-time recognition of different hand gestures.

### Gesture Recognition Results

![Hand Gesture Recognition Output](output/hand-gesture-output.png)

The system successfully identifies:

* **V Gesture**
* **Index Pointing Gesture**
* **L Right Gesture**

The detected hand is highlighted using a bounding box, while the predicted gesture name is displayed on the output frame.

---

## 📂 Project Structure

```text
hand-gesture-recognition/
│
├── README.md
├── hand_gesture_recognition.py
├── requirements.txt
│
├── output/
│   └── hand-gesture-output.png
│
├── dataset/
│
└── model/
```

> The exact structure may vary depending on the implementation and files used in the project.

---

## ▶️ How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/Rupasreesurapaneni777/Hand-Gesture-Recognition.git
```

### 2. Navigate to the Project Folder

```bash
cd Hand-Gesture-Recognition
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python hand_gesture_recognition.py
```

The webcam will start and the application will begin detecting hand gestures.

---

## 💡 Applications

Hand Gesture Recognition can be used in:

* Human-Computer Interaction
* Touchless Interfaces
* Sign Language Recognition
* Smart Home Control
* Virtual Reality
* Gaming
* Accessibility Applications
* Robot Control
* Gesture-Based Navigation

---

## 📈 Skills Demonstrated

Through this project, I gained practical experience with:

* Python programming
* OpenCV
* Computer Vision
* Machine Learning
* Image preprocessing
* Real-time video processing
* Feature extraction
* Gesture classification
* Webcam integration
* Debugging and testing

---

## 🔮 Future Improvements

Future enhancements can include:

* Recognizing more hand gestures
* Supporting both left and right hands
* Improving recognition accuracy
* Supporting dynamic hand gestures
* Recognizing complete sign-language alphabets
* Converting gestures into text
* Adding voice output
* Improving recognition under different lighting conditions
* Deploying the application with a user-friendly interface

---

## 👩‍💻 Author

**Surapaneni Rupa Sree**

AI & Data Science Graduate interested in:

* Data Analytics
* Python
* SQL
* Machine Learning
* Computer Vision
* Power BI

---

## ⭐ Support

If you find this project useful, consider giving the repository a **Star ⭐**.

Thank you for visiting my project!
