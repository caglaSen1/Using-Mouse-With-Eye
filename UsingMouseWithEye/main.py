import cv2
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)
# face_mesh = mp.solutions.face_mesh.FaceMesh()
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()   # to scale movement out with the screen

if not cap.isOpened():
    print("Fail to cap video")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)
    landmark_points = result.multi_face_landmarks
    frame_h, frame_w, _ch = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark  # bu satırda yüzlerin ilk tespit edilen yüzünün hatlarını landmarks değişkenine atar.
        for id, lm in enumerate(landmarks[474:478]):
            x = int(lm.x * frame_w)
            y = int(lm.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))
            if id == 1:  # we have 4 lm in an eye, we need to pick one of them
                screen_x = int(lm.x * screen_w)  # or screen_x = screen_w / frame_w * x
                screen_y = int(lm.y * screen_h)
                pyautogui.moveTo(screen_x, screen_y)

        left = [landmarks[145], landmarks[159]]
        for lm in left:
            x = int(lm.x * frame_w)
            y = int(lm.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))
        if (left[0].y - left[1].y) < 0.004:
            pyautogui.click()
            pyautogui.sleep(1)

    if ret:
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()