'''
시스템 구성 요소: YoLo, MediaPipe, Gemma(챗봇), STT/TTS
    Gemma: 상황에 맞는 텍스트를 생성 -> STT/TTS
    YoLo: 객체 탐지
    MediaPipe: 손모양 탐지

주제: 차량비서 AI
- Task: 
    YoLo기반 신호등 및 도로 객체 탐지
    * YoLo가 도로 내 신호등 탐지 > 탐지된 신호등만 크롭 > Gemma > Gemma: Instruction=너는 운전 보조 비서야 > 빨간색이면 멈춰/ 초록새이면 가/ 주황색이면: Gemma판단
    * 제약 - 실시간성?
    * LLM은 input으로 한 번에 개의 프레임을 하나의 프레임만 입력 받을 수 있음
        * 
    MediaPipe기반 자동차 기능 제어
    * 손동작 기반: 스피커 볼륨 조절 / 노래 재생, 정지 / 
'''
import cv2
import base64
from ultralytics import YOLO
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Gemma4ChatHandler

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

YOLO_MODEL_PATH = "src/models/YOLO/yolo11n_int8.engine"
GEMMA_MODEL_PATH = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
MMPROJ_PATH = "/home/test/VisionLLM/src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf"


CONTEXT_WINDOW = 2048
MAX_TOKENS = 20
DISPLAY_WIDTH = 600
DISPLAY_HEIGHT = 600

# 빨간색 마스크
def update_mask(image, h_upper1=10, h_lower2=170):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 100, 50], dtype=np.uint8)
    upper1 = np.array([h_upper1, 255, 255], dtype=np.uint8)
    lower2 = np.array([h_lower2, 100, 50], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv_image, lower1, upper1)
    mask2 = cv2.inRange(hsv_image, lower2, upper2)
    return cv2.bitwise_or(mask1, mask2)


# 중간 점 기준으로 각도 계산하는 함수
def calculate_angle(p1, p2, p3):
    vector1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    vector2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])

    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    cosine = np.dot(vector1, vector2) / (magnitude1 * magnitude2)
    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))


# base_option = python.BaseOptions(model_asset_path="src/models/MediaPipe/hand_landmarker.task")  # 모델 경로 지정하는 옵션
# options = vision.HandLandmarkerOptions(base_options=base_option, num_hands=2)                   # 모델 경로와 최대 손 개수 지정
# hand_detector = vision.HandLandmarker.create_from_options(options)                              # 해당 옵션으로 손 검출하는 객체 생성
# connections = vision.HandLandmarksConnections.HAND_CONNECTIONS                                  # 각 landmark를 잇는 연결선 정보

# finger_tips = (4, 8, 12, 16, 20)  # 손가락 끝 landmark의 index
# angle_threshold = 160             # 손가락 펼쳐짐 여부 판단 각도 임계값

# # 각 손가락의 각도를 계산할 landmark index
# finger_angle_points = (
#     (1, 2, 3),      # 엄지: 2번 중심
#     (5, 6, 7),      # 검지: 6번 중심
#     (9, 10, 11),    # 중지: 10번 중심
#     (13, 14, 15),   # 약지: 14번 중심
#     (17, 18, 19),   # 소지: 18번 중심
# )

# --모델 선언------------------------------------
yolo = YOLO(YOLO_MODEL_PATH)
chat_handler = Gemma4ChatHandler(clip_model_path=MMPROJ_PATH)
llm = Llama(
    model_path=GEMMA_MODEL_PATH,
    chat_handler=chat_handler,
    n_gpu_layers=-1,
    n_ctx=CONTEXT_WINDOW,
    n_batch=32,
    n_ubatch=32,
    verbose=False,
)

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

    # 프레임 높이와 너비
    h, w = frame.shape[:2]

    # 이미지 좌우 반전 및 RGB로 색공간 변환 (전처리)
    frame = cv2.flip(frame, 0)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # # 프레임 내 손 탐지
    # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    # result = hand_detector.detect(mp_image)
    # # 좌우 반전된 화면을 기준으로 왼손과 오른손 정보 변경
    # labels = ["Left" if handedness[0].category_name == "Right" else "Right" for handedness in result.handedness]

    # total_finger_count = 0
    # for hand in result.hand_landmarks:
    #     for point1_idx, point2_idx, point3_idx in finger_angle_points:
    #         angle = calculate_angle(
    #             hand[point1_idx],
    #             hand[point2_idx],
    #             hand[point3_idx],
    #         )
    #         if angle >= angle_threshold:
    #             total_finger_count += 1
            
    # if total_finger_count==1:
    #     pass # 소리 키워
    # if total_finger_count ==5:
    #     pass # 노래틀어
    # if total_finger_count == 0:
    #     pass # 노래꺼

            
                    
    # # 화면 좌측 상단에 손 개수와 펼친 손가락 개수 표시
    # cv2.putText(frame, f"Hands: {len(result.hand_landmarks)}", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    # cv2.putText(frame, f"Fingers: {total_finger_count}", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    # # 화면 우측 상단에 왼손/오른손/양손 여부 표시
    # handedness_text = " / ".join(labels)
    # text_size = cv2.getTextSize(handedness_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    # text_x = w - text_size[0] - 20
    # cv2.putText(frame, handedness_text, (text_x, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    # # 탐지 결과의 각 손마다 선과 점 그리기
    # for hand in result.hand_landmarks:
    #     h, w = frame.shape[:2]  # 프레임 높이와 너비
    #     points = [(int(p.x * w), int(p.y * h)) for p in hand]  # 프레임 높이와 너비 길이 기준 각 landmark 좌표

    #     # landmark를 연결하는 선 (skeleton) 그리기
    #     for c in connections:
    #         cv2.line(frame, points[c.start], points[c.end], (0, 255, 0), 2)

    #     # 각 관절 (landmark)에 점 그리기 (손가락 끝은 빨간 점, 그 외에는 파란 점)
    #     for i, point in enumerate(points):
    #         color = (0, 0, 255) if i in finger_tips else (255, 0, 0)
    #         cv2.circle(frame, point, 6 if i in finger_tips else 4, color, -1)

    results = yolo.predict(
        source=frame,   # source image
        conf=0.3,      # Confidence Threshold
        iou=0.5,        # IoU Threshold
        verbose=False,  # no output prints
        classes=[9],   # selected class
    )
    results = results[0]
    if results:
        # 신호등 Crop
        selected_box = results.boxes
        confidence = float(selected_box.conf[0].item())
        x1, y1, x2, y2 = (selected_box.xyxy[0].cpu().tolist())
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        traffic_light_image = frame[y1:y2, x1:x2].copy()
        # 이미지에서 Segmentation
        blurred_frame = cv2.GaussianBlur(traffic_light_image, (7, 7), 0)
        mask = update_mask(traffic_light_image)
        blurred_mask = update_mask(blurred_frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        eroded_mask = cv2.erode(blurred_mask, kernel, iterations=2)
        opened_mask = cv2.morphologyEx(blurred_mask, cv2.MORPH_OPEN, kernel)
        double_eroded_mask = cv2.erode(eroded_mask, kernel, iterations=2)

        # 프레임 캡쳐해서 저장 후 gemma한테 보여줄
        contours,_ = cv2.findContours(
            double_eroded_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 1:
            success, buffer = cv2.imencode(".jpg", traffic_light_image)

            if not success:
                raise RuntimeError("이미지 인코딩에 실패했습니다.")
            
            image_base64 = base64.b85encode(buffer).decode("utf-8")
            image_data = "data:image/jpeg;base64," + image_base64
            response = llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": """
                                    Instruction:
                                    주어진 신호등 이미지의 현재 색을 판단하시오.

                                    Constraint:
                                    반드시 다음 세 가지 중 하나만 대답하시오.
                                    빨간불
                                    노란불
                                    파란불

                                    다른 설명이나 문장을 추가하지 마시오.

                                    Output Format:
                                    한 단어.
                                """
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "현재 신호등의 색을 판단하시오."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data
                                },
                            },
                        ],
                    }
                ],
                max_tokens=MAX_TOKENS,
                temperature=0.0,
            )

            answer = response["choices"][0]["message"]["content"].strip()

            print("\n[Gemma]")
            print(answer)
        cv2.namedWindow("Traffic Light", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Traffic Light", DISPLAY_WIDTH, DISPLAY_HEIGHT)
        cv2.imshow("Traffic Light", double_eroded_mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    
    cv2.imshow("Traffic Light Detection", frame)

cap.release()
cv2.destroyAllWindows()

