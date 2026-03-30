import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

LABELS =  list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SAMPLES_PER_LABEL = 20

def extract_landmarks(hand_landmarks):
    landmarks = []
    wrist_x = hand_landmarks.landmark[0].x
    wrist_y = hand_landmarks.landmark[0].y

    for lm in hand_landmarks.landmark:
        landmarks.append(lm.x - wrist_x)
        landmarks.append(lm.y - wrist_y)
        
    return landmarks

def collect():
    cap = cv2.VideoCapture(0)
    all_data = []
    all_labels = []

    with mp_hands.Hands(
        static_image_mode = False,
        max_num_hands = 1,
        min_detection_confidence = 0.7
    ) as hands:
        
        for label in LABELS:
            # Wait for spacebar
            while True:
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f"Next: {label} - Press SPACE", (30, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2) 
                cv2.imshow("Collect Data", frame)
                key = cv2.waitKey(1)
                if key == 32: # spacebar
                    break

                count = 0
                while count < SAMPLES_PER_LABEL:
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = hands.process(rgb)
                
                    if result.multi_hand_landmarks:
                        hand = result.multi_hand_landmarks[0]
                        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                        features = extract_landmarks(hand)
                        all_data.append(features)
                        all_labels.append(label)
                        count += 1

    os.makedirs("data", exist_ok=True)
    with open("data/dataset.pkl", "wb") as f:
        pickle.dump({"data": np.array(all_data), "labels": all_labels}, f)