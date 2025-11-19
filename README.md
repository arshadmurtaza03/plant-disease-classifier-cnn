# 🌿 Plant Disease Classifier using CNN

A deep learning project that detects plant diseases from leaf images using Convolutional Neural Networks (CNN).

---
## Demo
[Live app (Streamlit)](https://arshadmurtaza03-plant-disease-classifier-cnn-app-lafnrl.streamlit.app/)

---

## 📋 Project Overview

This project uses a custom CNN model trained on the **New Plant Diseases Dataset** to classify plant leaf images into different disease categories. The model achieves high accuracy in identifying various plant diseases, making it useful for farmers and agricultural professionals.

## 🎯 Key Features

- **Custom CNN Architecture**: Built from scratch without transfer learning
- **Web Interface**: Simple Streamlit app for easy image upload and prediction
- **Multi-class Classification**: Identifies 38 different plant disease classes
- **Confidence Scores**: Displays prediction confidence for transparency
- **Predictions**: it Shows most likely disease

## 📊 Dataset

- **Dataset Name**: New Plant Diseases Dataset (Augmented)
- **Source**: [Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Classes**: 38 different plant disease categories
- **Total Images**: 87,000+ images

The dataset includes common diseases affecting plants like:
- Tomato diseases (Late Blight, Early Blight, etc.)
- Potato diseases
- Pepper diseases
- And many more

## 📂 Project Structure

```
plant-disease-classifier/
│
├── models/ # Saved model folder
│ ├── plant_disease_model.keras # Trained model file
│ └── class_names.json # List of disease classes
│
├── notebooks/
│ └── train_model.ipynb # Google Colab training notebook
│
├── app.py # Streamlit web application
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore # Git ignore file
```
## 🛠️ Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **Streamlit**: Web app framework
- **Python**: Programming language
- **NumPy**: Numerical computations
- **PIL**: Image processing
- **Google Colab**: Training environment (free T4 GPU)

## 📊 Model Architecture

The CNN model consists of:
- 5 Convolutional blocks (32, 64, 128, 256, 512 filters)
- Batch Normalization layers
- MaxPooling layers
- Dropout layers (0.5 and 0.3)
- Dense layers (512, 256 neurons)
- Output layer with softmax activation

**Input**: 128x128 RGB images
**Output**: 38 disease classes

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Kaggle account (for dataset download)
- Google Colab account (for training)

## 🛠️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/arshadmurtaza03/plant-disease-classifier-cnn.git
    cd plant-disease-classifier-cnn
    ```

2.  **Create and activate a virtual environment:**
    ```bash
        # Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

        # Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download the trained model**
- Train the model using `notebooks/train_model.ipynb` in Google Colab
- Download `plant_disease_model.keras` and `class_names.json`
- Place both files in the `models/` folder

    ### Training the Model

    - Open Google Colab
    - Upload the `notebooks/train_model.ipynb` notebook
    - Upload your `kaggle.json` file (download from Kaggle Account Settings)
    - Run all cells in the notebook
    - Download the trained model and class names file

    ### Running the App
    ```bash
    streamlit run app.py
    ```
    The app will open in your browser at `http://localhost:8501`


---

## 📈 Model Performance

- **Validation Accuracy**: **88.29%**
- **Number of Classes**: 38
- **Training Time**: ~50-60 minutes for **10** EPOCHS in Google Colab T4 GPU

## 📚 What I Learned
- **Deep Learning Basics** :Learned how CNNs extract image features using layers like convolution, pooling, and dense blocks.
- **Data Prep & Augmentation** : Understood how to clean, split, and augment image datasets for better model performance.
- **Cloud GPU Training** : Practiced training models on Google Colab using T4 GPUs and managing datasets through the Kaggle API.

## 🎓 Future Improvements
- Try transfer learning (ResNet, MobileNet)
- Add more plant species
- Improve UI/UX

## How to contact
- Author: Arshad Murtaza
- Email: arshadmurtaza2016@gmail.com
- LinkedIn: linkedin.com/in/arshadmurtaza

## 🙏 Acknowledgments

- Dataset provided by Kaggle
- TensorFlow and Streamlit communities
- Google Colab for free GPU access

---