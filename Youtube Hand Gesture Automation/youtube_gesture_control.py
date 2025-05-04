import cv2
import mediapipe as mp
import pyautogui
import time

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)
last_action_time = time.time()
cooldown = 1.5  # seconds

# To track previous wrist X position
prev_wrist_x = None

# Finger state detection
def fingers_up(lm):
    return [
        lm[4][0] > lm[3][0],                  # Thumb
        lm[8][1] < lm[6][1],                  # Index
        lm[12][1] < lm[10][1],                # Middle
        lm[16][1] < lm[14][1],                # Ring
        lm[20][1] < lm[18][1],                # Pinky
    ]

# Only one specific finger is up
def only_up(fingers, idx):
    return fingers[idx] and all(not fingers[i] for i in range(len(fingers)) if i != idx)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm = [(int(pt.x * w), int(pt.y * h)) for pt in hand_landmarks.landmark]
            fingers = fingers_up(lm)
            finger_count = sum(fingers)
            now = time.time()

            if now - last_action_time > cooldown:

                # 🔊 Volume Up — Any 3 fingers
                if finger_count == 3:
                    pyautogui.press('up')
                    print("Detected: 🔊 Volume Up (3 fingers)")
                    last_action_time = now

                # 🔉 Volume Down — Any 2 fingers
                elif finger_count == 2:
                    pyautogui.press('down')
                    print("Detected: 🔉 Volume Down (2 fingers)")
                    last_action_time = now

                # ⏯️ Pause/Play — Only pinky up
                elif only_up(fingers, 4):
                    pyautogui.press('space')
                    print("Detected: ⏯️ Pause/Play")
                    last_action_time = now

                # ⏩⏪ Next/Back — Open Palm + Swipe
                elif fingers == [True, True, True, True, True]:
                    wrist_x = lm[0][0]  # landmark 0 is wrist

                    if prev_wrist_x is not None:
                        movement = wrist_x - prev_wrist_x

                        # ⏩ Next — Swipe Right
                        if movement > 40:
                            pyautogui.press('right')
                            print("Detected: ⏩ Next (Swipe Right)")
                            last_action_time = now

                        # ⏪ Back — Swipe Left
                        elif movement < -40:
                            pyautogui.press('left')
                            print("Detected: ⏪ Back (Swipe Left)")
                            last_action_time = now

                    prev_wrist_x = wrist_x  # update for next frame

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("YouTube Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
