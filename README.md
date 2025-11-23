# 🚦 Traffic Sign Recognition (Deep Learning + Streamlit)

A lightweight **CNN-based Traffic Sign Classifier** built using **TensorFlow/Keras** and deployed through a modern **Streamlit UI**.  
Upload any traffic sign image and get instant predictions across **43 GTSRB classes**.

---

## 📌 Project Overview

This project uses a custom Convolutional Neural Network (CNN) to classify German traffic signs.  
The app provides a clean, responsive, dark-themed interface for fast, real-time inference.

---

## ⭐ Features

- ⚡ Real-time traffic sign prediction  
- 🎨 Modern dark-themed Streamlit UI  
- 🧠 Custom CNN model (trained on GTSRB)  
- 🖼 Upload JPG/PNG images  
- 📊 Top prediction with confidence score  
- ⚙️ Fast CPU inference  

---

## 🧠 Model Summary

- **Input:** 32 × 32 grayscale  
- **Architecture:**  
  - Conv2D → AvgPool  
  - Conv2D → AvgPool  
  - Dense(120)  
  - Dense(84)  
  - Dense(43 Softmax)  
- **Test Accuracy:** ~85%

---

## 📁 Project Structure

```
app.py
model.keras
classes.json
requirements.txt
TraficSignal.ipynb
```

---

## ⚙️ How It Works

1. User uploads a traffic sign image (JPG/PNG).  
2. Image is preprocessed (resize → grayscale → normalization).  
3. CNN model predicts probabilities for all 43 classes.  
4. Top class + confidence score is displayed instantly.  

---

## ▶️ Run Locally

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Start the Streamlit App
```bash
streamlit run app.py
```

### 3️⃣ Open in Browser
```
http://localhost:8501
```

---

## 📦 Dataset Source

This project is trained on the **GTSRB – German Traffic Sign Recognition Benchmark**.

Dataset Link:  
https://benchmark.ini.rub.de/gtsrb_news.html

---

## 📁 Model File

The `model.keras` file is included in the repository and loads automatically.  
Keep it in the project root folder for proper model inference.

---



## 🖼 Application Screenshot

<img width="1899" height="888" alt="image" src="https://github.com/user-attachments/assets/53e12ce0-fa5c-473d-8746-5ad639428828" />


## 🚀 Future Improvements

- Use Transfer Learning (ResNet / MobileNet)  
- Add real-time webcam prediction  
- Deploy on HuggingFace Spaces  
- Improve accuracy with augmentation  

---

## 👨‍💻 Author  
**Pallab Sharma**  
Data Analyst → AI/ML Practitioner

---

## ⭐ Support  
If you like this project, please ⭐ the repository!
