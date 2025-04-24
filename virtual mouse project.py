import cv2
import mediapipe as mp
import pyautogui

# Initialize video capture
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

# Get screen size
screen_width, screen_height = pyautogui.size()

# Initialize variables
index_y = 0

while True:
    # Read frame from webcam
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)

    hands = output.multi_hand_landmarks
    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark

            # Get coordinates of index finger tip
            index_finger = landmarks[8]
            x = int(index_finger.x * frame_width)
            y = int(index_finger.y * frame_height)

            # Convert coordinates to screen size
            screen_x = screen_width / frame_width * x
            screen_y = screen_height / frame_height * y

            # Move mouse
            pyautogui.moveTo(screen_x, screen_y)

            # Get coordinates of thumb tip
            thumb = landmarks[4]
            thumb_x = int(thumb.x * frame_width)
            thumb_y = int(thumb.y * frame_height)

            # Check if index finger and thumb are close to each other
            if abs(x - thumb_x) < 20 and abs(y - thumb_y) < 20:
                pyautogui.click()
                pyautogui.sleep(1)

    # Display the frame
    cv2.imshow("Virtual Mouse", frame)

    # Exit loop when 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()