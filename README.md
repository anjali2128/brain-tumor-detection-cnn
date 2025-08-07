🧠 Brain Tumor Detection using Deep Learning
This repository contains the source code and documentation for our B.Tech Project:
Leveraging Deep Learning for Automated Brain Tumor Classification from MRI Images
The system detects brain tumors in MRI scans using MobileNetV2 with Spatial Attention, aiming to assist in early diagnosis.

---

📑 Table of Contents

    📌 Project Overview
    🚀 Key Features
    📐 System Architecture
    🧠 Dataset
    ⚙️ Getting Started
    📊 Sample Results
    📈 Future Work
    📚 References
    🙌 Acknowledgements

  ---
  
📌 Project Overview

Early detection of brain tumors can significantly improve patient outcomes. Manual MRI analysis is labor-intensive and error-prone. This project utilizes Convolutional Neural Networks (CNN) with transfer learning (MobileNetV2) and spatial attention to automatically classify MRI images as:
✅ Normal (no tumor)
❌ Abnormal (tumor present)

---

🚀 Key Features

    🧠 End-to-end MRI classification (binary: tumor vs. no tumor)
    ⚡ MobileNetV2 for efficient feature extraction
    🔍 Spatial Attention layer for enhanced focus on tumor regions
    🧪 Data augmentation for robust generalization
    📊 Performance metrics: accuracy, confusion matrix, classification report
    🧩 Easily extendable to multi-class classification or segmentation

---
📐 System Architecture

MRI Image
   │
   ├──> Preprocessing & Augmentation
   │
   ├──> MobileNetV2 Backbone (Transfer Learning)
   │
   ├──> Spatial Attention Layer
   │
   ├──> Global Average Pooling
   │
   └──> Fully Connected Layers → Output (Normal / Abnormal)

---

⚙️ Getting Started
🧰 Installation

    git clone https://github.com/<your_username>/brain-tumor-mri-dl.git
    cd brain-tumor-mri-dl
    pip install -r requirements.txt

---

📦 Key Dependencies
  TensorFlow / Keras
  NumPy, Matplotlib, Seaborn
  scikit-learn

---
🧹 Data Preparation
Place your MRI images in the correct mri/ folder structure. Update any paths in train.py and evaluate.py if needed.

---

🧪 Training
   python train.py

    Uses data augmentation
    Trains MobileNetV2 + Spatial Attention
    Saves best weights (normal_abnormal_classifier.h5)
📈 Evaluation

    python evaluate.py
 Accuracy, loss, confusion matrix, classification report
---


🔍 Prediction on New MRI Image

    from tensorflow.keras.models import load_model
    model = load_model('normal_abnormal_classifier.h5', compile=False)
    
    # Preprocess your MRI image...
    pred = model.predict(image)
    print("Abnormal" if pred > 0.5 else "Normal")

---

📊 Sample Results
✅ Accuracy: Up to 98% on validation set
📉 Confusion Matrix & ROC Curve
📄 Classification Report with precision, recall, F1-score

----

📈 Future Work

      Multi-class classification (e.g., glioma, meningioma, pituitary)
      Tumor segmentation using U-Net
      Explainability via Grad-CAM
      Web/mobile deployment for clinicians
      Real-world validation on clinical MRI datasets

---

📚 References

      Khaliki, M.Z., Başarslan, M.S. (2024). Brain tumour detection from images using CNNs and transfer learning, Scientific Reports.
      Ben Brahim, S., et al. (2024). Brain Tumour Detection Using Deep CNNs, Applied Computational Intelligence and Soft Computing.
      Kaggle Brain MRI Dataset: https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection
🙌 Acknowledgements

    👨‍🏫 Supervisor: Dr. Jeevanajyothi Pujari (VIT, SCOPE)
    👩‍💻 Contributors: Anjali Mishra, Vaani Kohli, Srishti Sinha
    🏫 Institution: Vellore Institute of Technology (VIT)
