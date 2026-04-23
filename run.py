import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque, Counter

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def extract_landmarks(hand_landmarks):
    landmarks = []
    wrist_x = hand_landmarks.landmark[0].x
    wrist_y = hand_landmarks.landmark[0].y
    for lm in hand_landmarks.landmark:
        landmarks.append(lm.x - wrist_x)
        landmarks.append(lm.y - wrist_y)
    return landmarks


# Prediction buffer for smoothing
# deque(maxlen=5) is a fixed-size queue:
# when full and you add a new item, the oldest falls off automatically
prediction_buffer = deque(maxlen=5)

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6 # min_tracking_confidence controls when it gives up tracking and tries to detect fresh.
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        smoothed_label = ""
        confidence = 0.0

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # Extract features same normalization as training
            features = np.array(extract_landmarks(hand)).reshape(1, -1)
            # scikit-learn expects 2D array: (n_samples, n_features)
            # We have 1 sample with 42 features

            # Get prediction and confidence
            prediction = model.predict(features)[0]
            proba = model.predict_proba(features)[0]
            # predict_proba returns the fraction of trees that voted for each class
            # If 180 out of 200 trees voted for "A", confidence is 90%
            confidence = max(proba) * 100

            # Add to buffer and take majority vote
            prediction_buffer.append(prediction)
            smoothed_label = Counter(prediction_buffer).most_common(1)[0][0]
            # Counter counts occurrences of each item in the buffer
            # most_common(1) returns the most frequent item
            # [0][0] extracts the label (not the count)

        # Draw UI black rectangle behind text for readability
        cv2.rectangle(frame, (0, 0), (220, 100), (0, 0, 0), -1)
        cv2.putText(
            frame,
            smoothed_label,
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            3.0,
            (0, 255, 100),
            4
        )
        if confidence > 0:
            cv2.putText(
                frame,
                f"{confidence:.0f}%",
                (120, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (200, 200, 200),
                2
            )

        cv2.putText(
            frame,
            "Press Q to quit",
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (150, 150, 150),
            1
        )

        cv2.imshow("SignEase 2.0", frame)

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()