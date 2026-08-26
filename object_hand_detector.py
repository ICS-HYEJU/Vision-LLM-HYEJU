import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#공
green_lower = np.array([30, 60, 90], dtype=np.uint8)
green_upper = np.array([230, 115, 180], dtype=np.uint8)
# Morphological operation용 타원형 kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

#손
base_option = python.BaseOptions(model_asset_path="src/models/MediaPipe/hand_landmarker.task")  # 모델 경로 지정하는 옵션
options = vision.HandLandmarkerOptions(base_options=base_option, num_hands=2)                   # 모델 경로와 최대 손 개수 지정
hand_detector = vision.HandLandmarker.create_from_options(options)                              # 해당 옵션으로 손 검출하는 객체 생성
connections = vision.HandLandmarksConnections.HAND_CONNECTIONS                                  # 각 landmark를 잇는 연결선 정보
finger_tips = (4, 8, 12, 16, 20) 

pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

while True:
    ret, frame = cap.read()

    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # 이미지 좌우 반전 및 RGB로 색공간 변환 (전처리)
    frame = cv2.flip(frame, 0)

    # 공 탐지 전처리
    blr = cv2.GaussianBlur(frame, (11,11), 0)
    lab = cv2.cvtColor(blr, cv2.COLOR_BGR2LAB)

    # 손 탐지 전처리
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 공 탐지 마스크
    mask = cv2.inRange(lab, green_lower, green_upper)
        # Opening 2회, Dilation 2회
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # 프레임 내 공 탐지
    contour_lst, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 프레임 내 손 탐지
    result = hand_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    # 화면 좌측 상단에 텍스트 생성 (손 개수, 왼손/오른손/양손 여부)
    labels = ["Left" if h[0].category_name == "Right" else "Right" for h in result.handedness]
    cv2.putText(frame, f"Hands: {len(result.hand_landmarks)}", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    cv2.putText(frame, " / ".join(labels), (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)


    # 탐지 결과의 각 손마다 선과 점 그리기
    for hand in result.hand_landmarks:
        h,w = frame.shape[:2]
        points = [(int(p.x * w), int(p.y * h)) for p in hand]  # 프레임 높이와 너비 길이 기준 각 landmark 좌표

        # landmark를 연결하는 선 (skeleton) 그리기
        for c in connections:
            cv2.line(frame, points[c.start], points[c.end], (0,255,0), 2)

        # 각 관절 (landmark)에 점 그리기 (손가락 끝은 빨간 점, 그 외에는 파란 점)
        for i, point in enumerate(points):
            color = (0,0,255) if i in finger_tips else (255,0,0)
            cv2.circle(frame, point, 6 if i in finger_tips else 4, color, -1)
          
                # 공
            if len(contour_lst) > 0:
                    # 가장 큰 contour 선택
                    contour = max(contour_lst, key=cv2.contourArea)
                    # 최소 외접원 반지름
                    _, radius = cv2.minEnclosingCircle(contour)
                    # 무게중심
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    else:
                        center = (0, 0)
            
                    # 검출된 객체에 파란 원 overlay
                    cv2.circle(frame, center, int(radius), (255, 0, 0), 2)
                    cv2.circle(frame, center, 5, (255, 0, 0), -1)
    cv2.imshow("MediaPipe Hand Detection", frame)

hand_detector.close()
cap.release()
cv2.destroyAllWindows()


    