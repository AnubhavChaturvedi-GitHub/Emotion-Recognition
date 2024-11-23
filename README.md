# Emotion Recognition


## Overview

**Emotion Recognition** is a cutting-edge deep learning project designed to detect and classify human emotions based on facial expressions. Using a Convolutional Neural Network (CNN), the model is trained on the FER2013 dataset and can accurately recognize seven distinct emotions:  
**Angry**, **Disgust**, **Fear**, **Happy**, **Sad**, **Surprise**, and **Neutral**.

## Features

- **Real-Time Emotion Detection**: Analyze emotions live using webcam input.  
- **High Accuracy**: Powered by TensorFlow and trained on a robust dataset.  
- **User-Friendly Interface**: Simple, intuitive design for seamless interaction.  

## Technologies Used

- **Programming Language**: Python  
- **Libraries and Frameworks**:  
  - TensorFlow  
  - OpenCV  
  - NumPy  
  - Matplotlib  
  - Pandas  
  - scikit-learn  

---

## Installation

To get started with this project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/AnubhavChaturvedi-GitHub/Emotion-Recognition.git
   cd Emotion-Recognition
   ```

2. **Create a Virtual Environment** *(Optional but Recommended)*:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Train the Model
Train the CNN model on the FER2013 dataset:
```bash
python model_maker.py
```
After training, the model will be saved as **`emotion_model.h5`**.

### Run Real-Time Emotion Detection
Start the webcam-based emotion detection system:
```bash
python main.py
```
Press **`q`** to exit the webcam feed.

---

## Contributing

Contributions are welcome! If you have any ideas for improvements or new features, feel free to submit an issue or open a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- **FER2013 Dataset**: A publicly available dataset for emotion recognition.  
- **OpenCV**: For real-time video processing.  
- **TensorFlow**: For deep learning model training and inference.  

