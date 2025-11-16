Traffic Signal Prediction using Deep Learning

Project Overview
This project predicts traffic signal states (Red, Yellow, Green) using Deep Learning.
It helps optimize traffic flow, reduce congestion, and support intelligent transportation systems (ITS).
A Convolutional Neural Network (CNN) architecture is used to classify images or sensor inputs representing traffic conditions.

------------------------------------------------------------

Model Used
- Convolutional Neural Network (CNN)
- TensorFlow / Keras or PyTorch
- Softmax output layer for 3-class classification

------------------------------------------------------------

Project Structure
TrafficSignal_Prediction_DeepLearning
|
|-- dataset
|     |-- train
|     |-- test
|     |-- validation
|
|-- models
|     |-- traffic_signal_model.h5
|
|-- notebooks
|     |-- training_notebook.ipynb
|
|-- src
|     |-- predict.py
|
|-- images
|     |-- sample_predictions.png
|
|-- README.md

------------------------------------------------------------

Tech Stack
- Python
- TensorFlow / Keras (or PyTorch)
- OpenCV
- NumPy and Pandas
- Matplotlib and Seaborn

------------------------------------------------------------

Use Cases
- Smart City traffic regulation
- Automated Traffic Management Systems
- Road monitoring using cameras
- AI-powered traffic lights

------------------------------------------------------------

Training Details
- Image resizing: 64x64 or 128x128
- Augmentation: rotation, zoom, flip
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Accuracy: (Add your score here)

------------------------------------------------------------

Running Prediction
python src/predict.py --image sample.jpg

------------------------------------------------------------

Future Improvements
- Integrate YOLO for real-time detection
- Increase dataset size
- Add traffic density prediction
- Deploy using Flask or FastAPI

------------------------------------------------------------

Author
Pallab Sharma
Aspiring Data Scientist / Deep Learning Enthusiast
