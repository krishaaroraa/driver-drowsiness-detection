# Real-time Driver Drowsiness Detection using TensorFlow and OpenCV

A real-time driver drowsiness detection system designed to enhance road safety by continuously monitoring the driver’s facial movements. The system detects if the driver’s eyes remain closed for a prolonged duration and triggers an alarm to alert the driver, helping to prevent accidents caused by drowsiness.

---

## Dataset

The project uses the **Driver Drowsiness Detection** dataset from Kaggle. The dataset can be accessed [here](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd?resource=download-directory).

### Dataset Specifications:
- **Total Images**: 10,000
- **Categories**: Classified into 'Open' and 'Closed' eye states.
- **Training Set**: 8,000 images
- **Test Set**: 2,000 images
- **Purpose**: To train the model to detect whether the driver’s eyes are open or closed.

---

## OpenCV for Face Detection

OpenCV is used to detect the driver’s face in real time using the Haar Cascade classifier. By accurately identifying the facial region, the system focuses on the eyes, ensuring the drowsiness detection is efficient and precise.

---

## TensorFlow for Deep Learning Model

TensorFlow powers the deep learning model for eye state classification. The Convolutional Neural Network (CNN) is trained on the dataset to distinguish between 'Open' and 'Closed' eyes with high accuracy. The trained model integrates seamlessly with OpenCV to process frames in real time.

---

## Results

- **Model Accuracy**: Achieved a classification accuracy of 92.31% on the test set.
- **Detection Speed**: Processes video frames in real time (~37 FPS) on a standard CPU.
- **System Performance**: Successfully triggers an alarm when eyes remain closed for a specific duration. (3-4s)

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository/driver-drowsiness-detection.git
2. run drowsiness_detection.py
