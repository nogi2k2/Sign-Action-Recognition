# Military Sign Action Recognition App

A real-time sign recognition system that identifies military hand gestures such as **attention**, **pistol**, and **sniper** using Mediapipe Holistic, TensorFlow, and OpenCV. The model is trained on keypoint landmarks and visualized with dynamic probability bars

---

## 📋 Features

- 🧠 **Pose + Hand + Face Landmark Detection:** Uses Mediapipe Holistic for robust multi-part landmark extraction (33 pose, 468 face, 21 hand landmarks).
- 🔍 **Trained Gesture Classifier:** Trained a deep LSTM network for classifying signs like attention, pistol, and sniper based on 1662+ keypoints.
- 🖼️ **Probability Visualization:** Real-time bar chart overlaid on webcam feed to indicate classification confidence.
- 🎥 **Live Gesture Prediction:** Works directly with webcam feed to continuously predict and display gestures frame-by-frame.

---

## Tech Stack

- **Python 3.x** – Core application logic
- **TensorFlow/ Keras** – Framework for model architecture and training and model loading
- **OpenCV** – Webcam input and image handling
- **Mediapipe** – Landmark detection from webcam frames
- **NumPy** – Keypoint formatting and processing

---

## Project Structure

```

├── detector.h5                    # Pre-trained sign recognition model
├── script.py                      # Real-time webcam detection and prediction loop
├── Detection.ipynb                # Data collection, preprocessing, model building and training
├── MP_Data                        # Data collected for training deep LSTM network
|     |── attention
|     |── pistol
|     |── sniper
├── Logs                           # TensorBoard Logs

```
---

## 🧠 System Design & Pipeline

### Landmark Extraction

- Uses Mediapipe Holistic to extract
    - Pose landmarks (33 × 4 values)
    - Face mesh landmarks (468 × 3)
    - Left/Right hand landmarks (21 × 3 each)
    - Flattened into a single vector (1662 features) used as model input

### Prediction Logic

- Captures frames via webcam. Extracts landmarks and preprocesses them
- Buffers last 30 frames (for temporal understanding). Passes the sequence to the trained model. 
- Predicts gesture and overlays confidence bars

---

## 📆 Getting Started

### 📁 Prerequisites

- **Python 3.x**  
- **Tensorflow**
- **opencv-python**
- **mediapipe**
- **numpy**

---

###  🚀 Clone and Run

1. **Clone the repository**
```
$ git clone https://github.com/nogi2k2/Sign-Action-Recognition.git
```

2. **Navigate into the project directory**
```
$ cd <directory>
```

3. **Launch the app**  (model .h5 provided in repo)
```
python script.py
```

---
