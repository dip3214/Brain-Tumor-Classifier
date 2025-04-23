# README.md

This file is part of the modularized brain tumor classification project.

# README.md
This file is part of the modularized brain tumor classification project.

# 🧠 Brain Tumor Classification using CNN & Grad-CAM  
A Deep Learning-based Diagnostic Tool for MRI Scans


2. 📌 Overview / Introduction

This project uses a Convolutional Neural Network (CNN) with MobileNetV2 (transfer learning) to classify brain tumors from MRI scans into four categories:

- Glioma
- Meningioma
- Pituitary
- No Tumor

It also integrates Grad-CAM for visual interpretability — allowing users to see which areas of the scan influenced the model's predictions.

3. 🎯 Key Features:

- ✅ Transfer learning with MobileNetV2
- ✅ Handles class imbalance using class weights & data augmentation
- ✅ EarlyStopping and fine-tuning for better training
- ✅ Grad-CAM visualizations for interpretability
- ✅ Outputs predictions and confidence scores to CSV
- ✅ Clean, modular codebase for easy extension


4. 🧪 Model Architecture

**Base Model:** MobileNetV2 (pre-trained on ImageNet)

**Added Layers:**
- GlobalAveragePooling2D
- Dropout(0.3)
- Dense(4, activation='softmax')

5. 🧬 Dataset Structure

data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
├── Testing/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/

 Dataset not included due to size. You can download it from: https://www.kaggle.com/datasets/tombackert/brain-tumor-mri-data

 6. 🚀 How to Run the Project

 # 1. Clone the repository
git clone https://github.com/dip3214/Brain-Tumor-Classifier.git
cd Brain-Tumor-Classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place dataset in the /data/ folder (as shown above)

# 4. Run the pipeline
python main.py

7. 📊 Outputs
- Trained model: `outputs/brain_tumor_model.keras`
- CSV predictions: `outputs/prediction_results.csv`
- Grad-CAM heatmaps: `outputs/gradcam_outputs/`

8.  Grad-CAM Visualization Example:
![Grad-CAM Example](outputs/gradcam_outputs/sample_heatmap.png)

9.  Acknowledgements:
- Dataset: [https://www.kaggle.com/datasets/tombackert/brain-tumor-mri-data]
- Model: TensorFlow's MobileNetV2
- Research inspired by medical imaging AI applications

10. About the Author:
**Dipankar Mandal**  
MSc in Engineering Management | AI Researcher | Healthcare Tech Enthusiast  
University of York  
[LinkedIn](http://linkedin.com/in/dipankar-mandal3214) • [GitHub](https://github.com/dip3214) 


