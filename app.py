import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

# Load model and actions
model = load_model('action_detection_model.h5')

# MediaPipe utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Function to process image and extract results using MediaPipe
def mediapipe_detection(image: np.array, model) -> tuple:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Function to draw styled landmarks
def draw_styled_landmarks(image: np.array, results) -> None:
    specs = {
        'left': (121, 22, 76),
        'right': (245, 117, 66)
    }
    for hand, spec in specs.items():
        landmarks = getattr(results, f'{hand}_hand_landmarks')
        if landmarks:
            mp_drawing.draw_landmarks(
                image, landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=spec, thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=spec, thickness=2, circle_radius=2)
            )


actions = np.array(['Accident',  'Call',  'Doctor',  'Help', 'Hot', 'Lose', 'Pain', 'Thief'])

def run_action_detection():
    cap = cv2.VideoCapture(0)
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            # Prediction and visualization logic (simplified for brevity)

            cv2.imshow('OpenCV Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def app_main():
    st.title("Real-Time Action Detection")

    # WebRTC configuration to handle real-time video streams
    webrtc_ctx = webrtc_streamer(key="example", rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}))

    if webrtc_ctx.video_transformer:
        webrtc_ctx.video_transformer.model = model


class VideoTransformer:
    def __init__(self) -> None:
        self.model = None

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image, results = mediapipe_detection(image, self.model)
        draw_styled_landmarks(image, results)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    app_main()