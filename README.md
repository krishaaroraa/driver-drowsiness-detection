# 🚘 Driver Drowsiness Detection (Real-Time)

A real-time driver drowsiness detection system built using **TensorFlow** and **OpenCV**, designed to improve road safety by continuously monitoring the driver’s eye state.  
If the system detects that the driver's eyes remain closed for too long, it triggers an audible alarm to prevent accidents caused by drowsiness.

---

## ⭐ Features

- 🔍 **Real-time face and eye detection** using OpenCV Haar Cascade
- 🧠 **CNN-based eye state classifier** (Open vs Closed)
- 🚨 **Alarm system** activated when eyes stay closed for 3–4 seconds
- 🎥 Works with any webcam
- ⚡ Runs at ~37 FPS on CPU

---

## 📁 Dataset

This project uses the **Driver Drowsiness Detection Dataset** from Kaggle.

**Dataset Specs:**
- **Total Images:** 10,000  
- **Categories:**  
  - `Open`  
  - `Closed`  
- **Training Set:** 8,000 images  
- **Test Set:** 2,000 images  
- **Purpose:** Train a CNN to classify eye state with high accuracy

> **Link:** https://www.kaggle.com/datasets (Search: “Driver Drowsiness Detection”)

---

## 🧠 Deep Learning Model (TensorFlow)

A Convolutional Neural Network (CNN) is trained to classify eye images as:

- **Open**
- **Closed**

The trained model integrates with OpenCV to evaluate eye state from each video frame in real time.

---

## 🎯 Results

| Metric | Value |
|--------|--------|
| **Model Accuracy** | **92.31%** |
| **Real-Time FPS** | **~37 FPS** |
| **Alarm Trigger** | Closed-eye duration ≥ 3–4 seconds |

---

## 🎥 Real-Time Detection (OpenCV)

- Uses OpenCV’s Haar Cascade to detect face region  
- Focuses on eye ROI for classification  
- Blends TensorFlow predictions with frame-by-frame analysis  
- Activates alarm when drowsiness threshold is crossed

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/krishaaroraa/driver-drowsiness-detection.git
cd driver-drowsiness-detection
