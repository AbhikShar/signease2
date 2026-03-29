import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

LABELS =  list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")