import cv2
import mediapipe as mp
import time
from pythonosc import udp_client


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

osc_ip = "127.0.0.1"
osc_port = 8000
client = udp_client.SimpleUDPClient(osc_ip, osc_port)


def open_camera():
    for index in range(5):  
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  
        if cap.isOpened():
            print(f"[INFO] Camera {index} opened successfully.")
            return cap
        cap.release()
    print("[ERROR] No accessible camera found.")
    return None

cap = open_camera()
if cap is None:
    exit()


prev_time = 0
disconnect_count = 0

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    def get_landmark(landmarks, name):
        try:
            return landmarks[mp_pose.PoseLandmark[name].value]
        except KeyError:
            print(f"[WARNING] Invalid joint name: {name}")
            return None

    while True:
        if cap is None:
            print("[INFO] Attempting to reconnect camera...")
            cap = open_camera()
            if cap:
                time.sleep(2)
            continue

        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("[WARNING] Frame grab failed. Reinitializing camera...")
            disconnect_count += 1
            cap.release()
            cap = None
            time.sleep(2)
            if disconnect_count > 3:
                print("[ERROR] Too many failures. Waiting 5 seconds.")
                time.sleep(5)
                disconnect_count = 0
            continue
        else:
            disconnect_count = 0

        
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            lm = results.pose_landmarks.landmark
            joints_to_send = [
                'LEFT_WRIST', 'RIGHT_WRIST',
                'LEFT_SHOULDER', 'RIGHT_SHOULDER',
                'LEFT_KNEE', 'RIGHT_KNEE',
                'LEFT_HIP', 'RIGHT_HIP',
                'LEFT_ANKLE', 'RIGHT_ANKLE'
            ]

            for joint in joints_to_send:
                landmark = get_landmark(lm, joint)
                if landmark:
                    client.send_message(f"/{joint}", [landmark.x, landmark.y, landmark.z])

        
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        cv2.putText(image, f'FPS: {int(fps)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        
        cv2.imshow('Gesture Detection - Stable', image)

        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
