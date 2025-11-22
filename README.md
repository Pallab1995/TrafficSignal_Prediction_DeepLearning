Traffic Sign Recognition (Deep Learning + Streamlit)

Local App Link(streamlit):
http://localhost:8501

GitHub Repository:
https://github.com/Pallab1995/TrafficSignal_Prediction_DeepLearning

A lightweight CNN-based Traffic Sign Classifier built using TensorFlow/Keras and deployed through a modern Streamlit UI.
Upload any traffic sign image and get instant predictions across 43 GTSRB classes.

Features
- Real-time traffic sign prediction
- Modern dark-themed Streamlit UI
- Custom CNN model (trained on GTSRB)
- Upload JPG/PNG images
- Top prediction + confidence score
- Fast CPU inference

Model Summary
- Input: 32×32 grayscale image
- Architecture:
  Conv2D → AvgPool
  Conv2D → AvgPool
  Dense(120) → Dense(84) → Dense(43 Softmax)
- Test Accuracy: ~85%

Project Structure
app.py
model.keras
classes.json
requirements.txt
TraficSignal.ipynb

Run Locally
1. Install: pip install -r requirements.txt
2. Run: streamlit run app.py
3. Open: http://localhost:8501

Requirements
streamlit
tensorflow-cpu==2.12.0
numpy==1.26.4
pillow
h5py
matplotlib
seaborn
pandas
scikit-learn

Application Screenshot:

<img width="1899" height="888" alt="image" src="https://github.com/user-attachments/assets/53e12ce0-fa5c-473d-8746-5ad639428828" />


Author
Pallab Sharma
Data Analyst → AI/ML Practitioner
