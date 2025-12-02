# Brain Tumor Detection using Deep Learning

![Brain Tumor Detection](https://img.shields.io/badge/Brain-Tumor%20Detection-blue)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A deep learning model for detecting brain tumors from MRI scans. This project uses convolutional neural networks (CNNs) to classify brain MRI images into different categories: Glioma, Meningioma, No Tumor, and Pituitary tumor.

## 🚀 Features

- **Multi-class Classification**: Classifies brain MRI scans into 4 categories
- **High Accuracy**: Utilizes deep learning for precise tumor detection
- **User-friendly**: Simple command-line interface for predictions
- **Web Application**: Interactive web interface for easy usage (if applicable)

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Armankothariya/Brain-Tumor-Detection-System-.git
   cd Brain-Tumor-Detection-System-
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   If you don't have a requirements.txt, install the following packages:
   ```bash
   pip install tensorflow numpy pandas matplotlib scikit-learn opencv-python
   ```

## 🖥️ Usage

### Training the Model
```bash
python train_and_save_model.py
```

### Making Predictions
1. Place your MRI scan in the `test` directory
2. Run the prediction script:
   ```bash
   python predict.py --image path/to/your/mri_scan.jpg
   ```

## 🗂️ Project Structure

```
Brain-Tumor-Detection-System/
├── BrainTumorApp/         # Web application files (if applicable)
├── glioma/                # Training images - Glioma tumor
├── meningioma/            # Training images - Meningioma tumor
├── no_tumor/              # Training images - No tumor
├── pituitary/             # Training images - Pituitary tumor
├── test/                  # Test images
├── train/                 # Training data
├── .gitignore
├── README.md
└── train_and_save_model.py
```

## 🔧 Dependencies

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

## 🤖 How It Works

The model uses a Convolutional Neural Network (CNN) to analyze MRI scans and classify them into one of four categories:
1. Glioma Tumor
2. Meningioma Tumor
3. No Tumor
4. Pituitary Tumor

The training process involves:
1. Image preprocessing and augmentation
2. Building the CNN architecture
3. Training the model on the dataset
4. Evaluating performance
5. Saving the trained model

## 📊 Results

Model performance metrics:
- Accuracy: XX.XX%
- Precision: XX.XX%
- Recall: XX.XX%
- F1-Score: XX.XX%

(Note: Replace these with your actual model's performance metrics)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: [Brain Tumor Classification (MRI)](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri)
- Inspiration: Medical image analysis for better healthcare
- Special thanks to all contributors and open-source projects that made this possible
