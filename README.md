# 🧠 Brain Tumor Detection using Deep Learning

## 📌 Overview
This project implements a deep learning-based system for detecting brain tumors from MRI scans using Convolutional Neural Networks (CNNs) with transfer learning.

The model classifies MRI images into four categories:
- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

The goal of this project is to explore practical applications of deep learning in healthcare and understand challenges such as small datasets, class imbalance, and medical classification reliability.

---

## 🚀 Key Features
- Multi-class MRI image classification (4 tumor categories)
- Transfer learning using MobileNetV2
- Data augmentation for improved generalization
- End-to-end pipeline: preprocessing → training → evaluation → prediction
- Confusion matrix and detailed performance evaluation

---

## 🧠 Model Architecture
- Base Model: **MobileNetV2 (pre-trained on ImageNet)**
- Frozen base layers for feature extraction
- Custom classifier:
  - Global Average Pooling
  - Dense layer (256 units, ReLU)
  - Dropout (0.3)
  - Output layer (Softmax, 4 classes)

- Total Parameters: **2.58M**
- Trainable Parameters: **~329K (12.7%)**

---

## 📊 Dataset
The dataset consists of MRI images categorized into:
- Glioma (80 images)
- Meningioma (63 images)
- Pituitary (54 images)
- No Tumor (49 images)

Split:
- Training: 196 images  
- Testing: 50 images  

⚠️ Note: Dataset size is relatively small, which impacts generalization.

---

## 📈 Model Performance

### 🔥 Overall Metrics
- **Test Accuracy: 74.00%**
- **Total Test Samples: 50**

### 📊 Per-Class Performance

| Class        | Accuracy | Precision | Recall | F1-Score |
|-------------|---------|----------|--------|----------|
| Glioma      | 81.25%  | 0.81     | 0.81   | 0.81     |
| Meningioma  | 53.85%  | 0.70     | 0.54   | 0.61     |
| No Tumor    | 80.00%  | 0.89     | 0.80   | 0.84     |
| Pituitary   | 81.82%  | 0.60     | 0.82   | 0.69     |

📌 **Observation:**
- Strong performance on Glioma, Pituitary, and No Tumor classes
- Weak performance on Meningioma due to class confusion

---

## 🔍 Key Insights
- The model performs well on most tumor types (≈80% accuracy)
- Meningioma classification is challenging (low recall: 0.54)
- Model shows moderate confidence (~70–80%) across predictions
- High parameter-to-data ratio (~13K params/sample) suggests overfitting risk

---

## ⚠️ Limitations
- Small dataset size (196 training samples)
- Class imbalance across tumor types
- Limited generalization to real-world clinical data
- Not suitable for clinical deployment

---

## 🛠️ Installation

```bash
git clone https://github.com/Armankothariya/Brain-Tumor-Detection-System-.git
cd Brain-Tumor-Detection-System-


python -m venv venv

Activate:

Windows:
.\venv\Scripts\activate
Linux/Mac:
source venv/bin/activate
pip install -r requirements.txt
🖥️ Usage
Train the model
python train_and_save_model.py
Evaluate model
Generates classification report & confusion matrix
Predict on new image
python predict.py --image path/to/image.jpg
📁 Project Structure
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
🔮 Future Improvements
Increase dataset size (1000+ samples)
Fine-tune MobileNetV2 layers
Try advanced models (ResNet, EfficientNet)
Improve meningioma classification
Add Grad-CAM for model interpretability
Deploy as web-based diagnostic tool
⚕️ Clinical Readiness

❌ Not ready for clinical deployment

Reasons:

Low recall for certain tumor types
Limited dataset size
No validation on external datasets
🤝 Contributions

Contributions are welcome! Feel free to fork and improve the project.

📄 License

MIT License

🙌 Acknowledgment

This project explores the intersection of deep learning and healthcare, focusing on real-world challenges in medical image classification.
