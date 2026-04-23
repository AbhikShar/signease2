import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

# Mediapipe hand module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Store 26 letters
LABELS =  list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SAMPLES_PER_LABEL = 100

def extract_landmarks(hand_landmarks):
    landmarks = []

    # store the reference point (wrist)
    wrist_x = hand_landmarks.landmark[0].x
    wrist_y = hand_landmarks.landmark[0].y

    for lm in hand_landmarks.landmark:
        # Subtract wrist position from each landmark
        landmarks.append(lm.x - wrist_x)
        landmarks.append(lm.y - wrist_y)
        
    # normalized 42 number feature vector with respect to wrist
    return landmarks

def collect():
    cap = cv2.VideoCapture(0) # 0 = first camera
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    all_data = [] # will hold feature vectors
    all_labels = [] # will hold corrensponding letters

    with mp_hands.Hands(
        static_image_mode = False, # video mode
        max_num_hands = 1, 
        min_detection_confidence = 0.7 # only accept if MediaPipe is 70%+ confident
    ) as hands:
        
        for label in LABELS:
            print(f"\nGet read for letter: {label}")
            print("Press SPACE to start, Q to quit")

            # Wait for spacebar
            while True:
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1) # makes vid more intuitive like a mirror
                cv2.putText(frame, 
                            f"Next: {label} - Press SPACE", 
                            (30, 50), 
                            cv2.FONT_HERSHEY_COMPLEX, 
                            0.8, 
                            (0, 255, 0), 
                            2) 
                cv2.imshow("Collect Data", frame) # show image in a new window
                key = cv2.waitKey(1) # always call waitKey() after imshow()
                if key == 32: # spacebar
                    break
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            #Collect SAMPLES_PER_LABEL frames for this letter
            count = 0
            while count < SAMPLES_PER_LABEL:
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)

                # required BGR to RGB conversion
                # OpenCV reads frames in BGR, Mediapipe expects RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb) # powerful mediapipe command
                
                if result.multi_hand_landmarks: # only save samples when Hand is detected
                    hand = result.multi_hand_landmarks[0] # [0] -> first hand detected
                    mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                    features = extract_landmarks(hand)
                    all_data.append(features)
                    all_labels.append(label)
                    count += 1
                    # Show progress
                cv2.putText(
                        frame,
                        f"{label}: {count}/{SAMPLES_PER_LABEL}",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 200, 255),
                        2
                    )
                cv2.imshow("Collect Data", frame)
                cv2.waitKey(1)

            print(f"Done: {label} ({count} samples)")

    cap.release()
    cv2.destroyAllWindows()

    # Save the dataset to disk using pickle
    # Save both arrays in one dictionary so they stay linked
    os.makedirs("data", exist_ok=True)
    with open("data/dataset.pkl", "wb") as f:
        pickle.dump({
            "data": np.array(all_data), 
            "labels": all_labels}, 
            f)
    
    print(f"\nSaved {len(all_data)} samples to data/dataset.pkl")
if __name__ == "__main__":
    collect()