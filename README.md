# 🧠 Brain Tumor Detection using Deep Learning

<p align="center">
  <img src="https://img.icons8.com/fluency/96/brain.png" width="100"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow"/>
  <img src="https://img.shields.io/badge/Model-MobileNetV2-green"/>
  <img src="https://img.shields.io/badge/Accuracy-74%25-brightgreen"/>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey"/>
</p>

---

## 📌 Overview
This project implements a deep learning-based system for detecting brain tumors from MRI scans using Convolutional Neural Networks (CNNs) with transfer learning.

The model classifies MRI images into:
- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

---

## 🚀 Features
- Multi-class MRI classification (4 classes)
- Transfer learning with MobileNetV2
- Data augmentation for better generalization
- End-to-end ML pipeline
- Performance evaluation with confusion matrix

---

## 🧠 Model Architecture
- MobileNetV2 (pre-trained)
- Global Average Pooling
- Dense (256) + Dropout
- Softmax output layer

---

## 📊 Model Performance

### 🔥 Overall
- Accuracy: **74%**
- Test Samples: **50**

### 📊 Class-wise Performance

| Class | Accuracy |
|------|---------|
| Glioma | 81.25% |
| Meningioma | 53.85% |
| No Tumor | 80.00% |
| Pituitary | 81.82% |

---

## 📸 Results Visualization

### Confusion Matrix
<p align="center">
  <img src="https://github.com/user-attachments/assets/169ae4ca-1895-4596-a96d-a030345697e9" width="750"/>
</p>


---

## ⚙️ Installation

```bash
git clone https://github.com/Armankothariya/Brain-Tumor-Detection-System-.git
cd Brain-Tumor-Detection-System-
```

```bash
python -m venv venv
```

### Activate environment

**Windows:**
```bash
.\venv\Scripts\activate
```

**Linux / Mac:**
```bash
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

---

## 🖥️ Usage

### Train model
```bash
python train_and_save_model.py
```

### Predict
```bash
python predict.py --image path/to/image.jpg
```

---

## 📁 Project Structure

```bash
Brain-Tumor-Detection-System/
│── glioma/
│── meningioma/
│── no_tumor/
│── pituitary/
│── train/
│── test/
│── train_and_save_model.py
│── requirements.txt
│── README.md
```

---

## 🔮 Future Improvements
- Increase dataset size
- Improve meningioma accuracy
- Use ResNet / EfficientNet
- Add Grad-CAM visualization
- Deploy as web app

---

## ⚕️ Clinical Readiness

❌ Not ready for clinical use

**Reasons:**
- Limited dataset
- Low recall for meningioma
- No external validation

---

## 🤝 Contributing
Pull requests are welcome!

---

## 📄 License
MIT License

---

## 🙌 Acknowledgment
Focused on applying AI in healthcare and solving real-world medical challenges.
