#  🧠 Action Detection Model 🎬

A deep learning model built using **MediaPipe**, **TensorFlow**, and **OpenCV** to detect and classify human actions in real-time through webcam or video input.

## 🚀 Features

- Real-time pose estimation using **MediaPipe**
- Action classification using a trained LSTM model
- Visual feedback with OpenCV overlay
- Modular code for training and testing
- Works on live webcam feed and pre-recorded videos

---

## 🖥️ Tech Stack

- **Frontend (Visualization)**: OpenCV
- **ML Libraries**: TensorFlow, NumPy, MediaPipe
- **Backend**: Python
- **Notebook**: Jupyter Notebook
- **Testing**: Matplotlib, NumPy
- **Deployment**: Local machine
## 📁 Project Structure

Action-Detection-Model/
├── Action Detection model.ipynb # Jupyter notebook for training/testing
├── model.h5 / action.keras # Trained model
├── 0.npy # Extracted pose keypoints
├── data/ # Raw and processed data
├── Logs/ # TensorBoard logs
├── utils.py # Utility functions (if any)
├── README.md # You're here!

yaml
Copy
Edit
🧠 Model Training
Pose landmarks are extracted using MediaPipe

Sequences of frames are converted to NumPy arrays

LSTM model is trained to classify actions like walking, waving, etc.

Saved using .h5 or .keras formats

