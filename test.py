import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3
import os
import sys

# ------------------ Initialization ------------------
print("✅ Starting Sign Language Recognition...")

# Check model files
if not os.path.exists("Model/keras_model.h5") or not os.path.exists("Model/labels.txt"):
    print("❌ Model files not found in 'Model/' directory.")
    sys.exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot access the webcam.")
    sys.exit()

detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
engine = pyttsx3.init()

# Settings
offset = 20
imgSize = 300
word = ""

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
          "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "Space"]

print("🎥 Webcam initialized. Press 's' to add letter, 't' to speak, 'q' to quit.")

# ------------------ Main Loop ------------------
while True:
    success, img = cap.read()
    if not success:
        print("❌ Failed to read frame from webcam.")
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    current_label = ""

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        current_label = labels[index].strip().lower()

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 120, y - offset), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, current_label.upper(), (x, y - 26),
                    cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

    # Draw word box
    cv2.rectangle(imgOutput, (20, 10), (1100, 80), (0, 0, 255), cv2.FILLED)
    cv2.putText(imgOutput, f"Word: {word}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # Show the image
    cv2.imshow("Image", imgOutput)

    key = cv2.waitKey(1)

    # Add current letter
    if key == ord('s') and hands:
        if current_label == "space":
            word += " "
            print("➕ Added SPACE")
        elif current_label != "":
            word += current_label.upper()
            print(f"➕ Added letter: {current_label.upper()}")

    # Clear word
    if key == ord('c'):
        word = ""
        print("🧹 Cleared word")

    # Backspace
    if key == ord('d'):
        if word:
            word = word[:-1]
            print("⬅️ Removed last character")

    # Speak word
    if key == ord('t'):
        if word.strip():
            print("🔊 Speaking:", word.strip())
            engine.say(word.strip())
            engine.runAndWait()

    # Quit
    if key == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
