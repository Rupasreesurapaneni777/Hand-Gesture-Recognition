# ✋ Hand Gesture Recognition System

A real-time **Hand Gesture Recognition System** developed using **Python, OpenCV, NumPy, and Computer Vision**.

The application captures live video from a webcam, detects the hand region, identifies finger and palm positions, and recognizes predefined hand gestures such as **V**, **L Right**, and **Index Pointing**.

---

## 📌 Project Overview

The **Hand Gesture Recognition System** is a Computer Vision project designed to detect and recognize hand gestures from live webcam input.

The system uses image-processing techniques such as:

* Background subtraction
* HSV color processing
* Histogram back projection
* Thresholding
* Contour detection
* Convex hull detection
* Palm-center detection
* Finger-position detection
* Gesture comparison

After detecting the hand and fingers, the system compares the detected pattern with predefined gestures and displays the recognized gesture in real time.

---

## 🎯 Objective

The main objectives of this project are:

* Capture real-time video using a webcam
* Detect the hand from the background
* Identify palm and finger positions
* Recognize predefined hand gestures
* Display the detected gesture in real time
* Demonstrate touchless Human-Computer Interaction

---

## 🚀 Features

* 📷 Real-time webcam input
* ✋ Hand detection
* 🖐️ Finger detection
* 🔵 Palm-center detection
* 📐 Finger-angle calculation
* 🔍 Contour and convex hull detection
* 🎯 Region of Interest for hand processing
* 🤖 Gesture recognition
* ⚡ Real-time processing
* 🔄 Reset option
* ⌨️ Simple keyboard controls

---

## 🛠️ Technologies Used

* **Python**
* **OpenCV**
* **NumPy**
* **Computer Vision**
* **Image Processing**
* **VS Code**

---

## ⚙️ How the System Works

The project follows the workflow below:

```text
Webcam Input
      ↓
Background Capture
      ↓
Hand Histogram Capture
      ↓
Background Removal
      ↓
HSV Conversion
      ↓
Thresholding
      ↓
Contour Detection
      ↓
Convex Hull Detection
      ↓
Palm Center Detection
      ↓
Finger Detection
      ↓
Gesture Comparison
      ↓
Recognized Gesture
```

### 1. Webcam Input

The application accesses the system webcam using OpenCV and continuously captures live video frames.

### 2. Background Capture

Before detecting the hand, the application captures the background.

The user should keep the hand outside the detection area and press:

```text
b
```

This creates a background model that helps separate the hand from the surroundings.

### 3. Hand Histogram Capture

After capturing the background, the user places the hand over the small boxes displayed on the screen and presses:

```text
c
```

The application captures the color information of the hand and creates a histogram.

### 4. Background Removal

The stored background model is used to remove unnecessary background information from each video frame.

### 5. Hand Detection

The application converts the image into HSV color space and applies histogram back projection, filtering, and thresholding to detect the hand region.

### 6. Contour Detection

The system finds contours from the processed image and selects the largest contour as the detected hand.

### 7. Palm Detection

The palm center and palm radius are calculated from the detected hand contour.

### 8. Finger Detection

The system analyzes points from the convex hull and identifies possible finger positions.

### 9. Gesture Recognition

The detected finger positions, angles, and palm information are compared with predefined gesture patterns.

The final recognized gesture is displayed on the webcam output.

---

## ✋ Supported Gestures

The current implementation contains the following predefined gestures:

| Hand Gesture             | Recognized Output |
| ------------------------ | ----------------- |
| ✌️ Index + Middle Finger | `V`               |
| 👆 L-Shaped Hand Gesture | `L_right`         |
| ☝️ Index Finger Pointing | `Index_Pointing`  |

The predefined gesture patterns are stored in:

```text
GestureAPI.py
```

---

## 📸 Project Output

The following screenshot shows the **Hand Gesture Recognition System running with real-time webcam input**.

### Real-Time Hand Gesture Detection

![Hand Gesture Recognition Output](output/hand-gesture-output.png)

The output window displays the live webcam feed along with the region used for hand detection and processing.

The system continuously processes the camera frames and analyzes the hand placed inside the detection area.

---

## ⌨️ Controls

The application provides simple keyboard controls:

| Key | Action                 |
| --- | ---------------------- |
| `b` | Capture background     |
| `c` | Capture hand histogram |
| `r` | Reset the system       |
| `q` | Quit the application   |

---

## 📂 Project Structure

```text
hand-gesture-recognition-sytem/
│
├── GestureAPI.py
├── HandRecognition.py
├── requirements.txt
├── README.md
├── LICENSE
│
├── output/
│   └── hand-gesture-output.png
│
└── docs/
```

### File Description

**HandRecognition.py**

Main application file responsible for:

* Accessing webcam
* Image processing
* Background removal
* Hand detection
* Finger detection
* Gesture recognition
* Displaying the output

**GestureAPI.py**

Contains:

* Gesture class
* Predefined gesture patterns
* Finger-angle calculations
* Gesture comparison logic
* Gesture decision logic

**requirements.txt**

Contains the Python libraries required to run the project.

---

## 📦 Installation

### 1. Download or Clone the Project

Download the repository and open the project folder in VS Code.

### 2. Open Terminal

Navigate to the folder containing:

```text
HandRecognition.py
GestureAPI.py
requirements.txt
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

For Windows:

```bash
.venv\Scripts\activate
```

### 5. Install Required Libraries

```bash
python -m pip install -r requirements.txt
```

The main libraries required are:

```text
numpy
opencv-python
```

---

## ▶️ How to Run the Project

Run the following command:

```bash
python HandRecognition.py
```

The webcam window will open.

### Step 1

Remove your hand from the camera detection area and press:

```text
b
```

### Step 2

Place your hand over the small boxes displayed on the screen and press:

```text
c
```

### Step 3

Place your hand inside the detection region and perform one of the supported gestures.

The system will analyze the hand and display the recognized gesture.

### Step 4

Press:

```text
q
```

to close the application.

---

## 🧠 Computer Vision Techniques Used

### Background Subtraction

Background subtraction separates the moving hand from the background.

### HSV Color Space

The captured image is converted from BGR to HSV color space to improve hand-color processing.

### Histogram Back Projection

The captured hand histogram is used to locate similar hand-color regions in each video frame.

### Thresholding

The processed hand image is converted into a binary image to make contour detection easier.

### Contour Detection

The largest detected contour is treated as the hand.

### Convex Hull

A convex hull is generated around the hand contour to identify possible fingertips.

### Palm Center Detection

The system calculates an approximate center and radius of the palm.

### Finger Detection

Potential fingertip positions are filtered based on their distance from the palm center.

### Gesture Matching

Finger positions and angles are compared with predefined gesture patterns to determine the final gesture.

---

## 💡 Applications

Hand Gesture Recognition can be useful in:

* Human-Computer Interaction
* Touchless Interfaces
* Gesture-Controlled Applications
* Gaming
* Smart Home Control
* Robot Control
* Accessibility Systems
* Virtual Reality
* Interactive Computer Vision Applications

---

## 📈 Skills Demonstrated

This project demonstrates practical knowledge of:

* Python Programming
* OpenCV
* NumPy
* Computer Vision
* Real-Time Video Processing
* Image Preprocessing
* Background Subtraction
* Histogram Processing
* Contour Detection
* Convex Hull
* Feature Extraction
* Gesture Recognition
* Webcam Integration
* Debugging

---

## 🔮 Future Improvements

Future improvements can include:

* Recognizing more hand gestures
* Supporting both left and right hands
* Improving recognition under different lighting conditions
* Detecting dynamic gestures
* Adding sign-language alphabet recognition
* Converting recognized gestures into text
* Adding voice output
* Improving gesture accuracy
* Creating a user-friendly graphical interface
* Using deep learning for advanced gesture recognition

---

## ✅ Result

The project successfully demonstrates a real-time **Hand Gesture Recognition System** using Python and OpenCV.

The application can:

* Capture live webcam input
* Separate the hand from the background
* Detect the palm and fingers
* Analyze hand geometry
* Compare detected patterns with predefined gestures
* Recognize supported hand gestures in real time

---

## 👩‍💻 Author

**Surapaneni Rupa Sree**

Interested in:

* Data Analytics
* Python
* SQL
* Machine Learning
* Computer Vision
* Power BI

---

## 📄 License

This project is licensed under the **MIT License**.

See the `LICENSE` file for more information.


## ⭐ Support

If you find this project useful, consider giving the repository a **Star ⭐**.

Thank you for visiting my project!
