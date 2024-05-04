# Life-Saving Signs: Real-Time Detection of Emergency Indian Sign Language

In emergency situations, every second counts, especially for the hearing-impaired community who rely on sign language for communication. "Life-Saving Signs" is a groundbreaking project that employs advanced machine learning techniques to recognize and interpret critical emergency gestures in Indian Sign Language (ISL) in real time. The core objective of this initiative is to bridge communication gaps and enhance accessibility, particularly in urgent scenarios where traditional communication methods might fail or be unavailable.

This project utilizes Long Short-Term Memory (LSTM) networks, a type of recurrent neural network suited to classifying, processing, and predicting sequences, in conjunction with MediaPipe, a versatile framework for building multimodal (audio, video, etc.) applied machine learning pipelines. By focusing on a specific set of ISL emergency signs — including 'Accident', 'Call', 'Doctor', 'Help', 'Hot', 'Lose', 'Pain', and 'Thief' — the system can swiftly recognize and respond to the critical needs of ISL users, facilitating faster response times in emergency situations.

Development stages encompass initial data collection via webcams, where gestures are recorded and used to train the LSTM model. This is followed by refining the model through rigorous testing to ensure accuracy and responsiveness. The final product is deployed using a Streamlit web application, allowing for easy interaction and scalability.

"Life-Saving Signs" is not just a technological advancement; it is a step towards inclusive communication, ensuring that the deaf and hard-of-hearing community can easily summon help and communicate distress in critical situations without barriers.

## Repository Contents

- `notebooks/`: Jupyter notebooks with step-by-step instructions.
  - `1_online_data_ISL_detection.ipynb`: Explores the possibility of training an LSTM model using a dataset collected online.
  - `2_own_data_ISL_detection.ipynb`: Demonstrates data collection using a webcam and training an LSTM model after extracting landmarks from the face, pose, and both hands.
  - `3_hands_only_isl_detection.ipynb`: Focuses on collecting and training data using only both hand landmarks, and implements an interface for this model.
- `src/`: Source code for the application.
  - `app.py`: Streamlit application for real-time action detection.
  - `action_detection_model.h5`: Pre-trained model.
- `README.md`: This file.

## Installation

Follow these steps to set up the project:

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/indian-sign-language-detection.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Collection

Navigate to the `notebooks/` directory and open the respective notebook to collect data:

```bash
jupyter notebook 1_online_data_ISL_detection.ipynb
```

Follow the instructions within the notebook to collect and process data.

### Model Training

To train the LSTM model, open:

```bash
jupyter notebook 2_own_data_ISL_detection.ipynb
```

This notebook guides you through the model training process using the collected data.

### Deployment

Run the Streamlit application by navigating to the `src/` directory and executing:

```bash
streamlit run app.py
```

This command starts a local web server and opens the Streamlit application in your default web browser.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.
