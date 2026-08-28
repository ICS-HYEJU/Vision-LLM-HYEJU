'''
시스템 구성 요소: YOLO, MediaPipe, Gemma(챗봇), STT/TTS

주제: 차량 비서 AI
- YOLO 기반 신호등 및 도로 객체 탐지
- MediaPipe 기반 손동작 자동차 기능 제어
- Gemma 기반 운전 보조 응답 생성
'''

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Gemma4ChatHandler
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


YOLO_MODEL_PATH = "src/models/YOLO/yolo11n_int8.engine"
GEMMA_MODEL_PATH = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
MMPROJ_PATH = "src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf"

CONTEXT_WINDOW = 2048
DISPLAY_WIDTH = 400
DISPLAY_HEIGHT = 400

# COCO의 traffic light(class 9)는 작은 객체라 기본값(0.25)보다 낮게 사용하되,
# 기존 0.015처럼 지나치게 낮춰 오탐이 많아지지 않도록 한다.
YOLO_CONFIDENCE = 0.08
YOLO_IOU = 0.50
BBOX_PADDING_RATIO = 0.20
BBOX_SMOOTHING_ALPHA = 0.55


def calculate_angle(p1, p2, p3):
    """p2를 꼭짓점으로 하는 세 점의 각도를 계산한다."""
    vector1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    vector2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])

    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    cosine = np.dot(vector1, vector2) / (magnitude1 * magnitude2)
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def select_traffic_light(boxes, frame_width, frame_height):
    """여러 신호등 중 주행 방향에 있을 가능성이 높은 박스를 선택한다.

    단순히 화면에서 가장 위에 있는 박스를 선택하면 작은 오탐이 자주 선택된다.
    따라서 YOLO 신뢰도를 가장 크게 반영하고, 화면 중앙 및 상단에 가까울수록
    약간의 가산점을 부여한다.
    """
    candidates = []
    frame_area = float(frame_width * frame_height)

    for box in boxes:
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().tolist())
        box_width = x2 - x1
        box_height = y2 - y1

        if box_width < 2 or box_height < 2:
            continue

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        center_score = max(
            0.0,
            1.0 - abs(center_x - frame_width / 2.0) / (frame_width / 2.0),
        )
        upper_score = max(0.0, 1.0 - center_y / frame_height)
        area_ratio = (box_width * box_height) / frame_area
        size_score = min(area_ratio / 0.01, 1.0)

        selection_score = (
            0.70 * confidence
            + 0.15 * center_score
            + 0.10 * upper_score
            + 0.05 * size_score
        )
        candidates.append((selection_score, confidence, (x1, y1, x2, y2)))

    if not candidates:
        return None

    _, confidence, coordinates = max(candidates, key=lambda candidate: candidate[0])
    return coordinates, confidence


def smooth_box(previous_box, current_box, alpha=BBOX_SMOOTHING_ALPHA):
    """프레임 사이의 작은 박스 떨림을 지수 이동 평균으로 완화한다."""
    current = np.asarray(current_box, dtype=np.float32)
    if previous_box is None:
        return current
    previous = np.asarray(previous_box, dtype=np.float32)
    return alpha * current + (1.0 - alpha) * previous


def clamp_and_pad_box(box, frame_width, frame_height):
    """Crop에 신호등 주변 문맥이 조금 포함되도록 박스를 확장한다."""
    x1, y1, x2, y2 = box
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    pad_x = box_width * BBOX_PADDING_RATIO
    pad_y = box_height * BBOX_PADDING_RATIO

    return (
        max(0, int(round(x1 - pad_x))),
        max(0, int(round(y1 - pad_y))),
        min(frame_width, int(round(x2 + pad_x))),
        min(frame_height, int(round(y2 + pad_y))),
    )


# MediaPipe 손 검출기
base_option = python.BaseOptions(
    model_asset_path="src/models/MediaPipe/hand_landmarker.task"
)
options = vision.HandLandmarkerOptions(base_options=base_option, num_hands=2)
hand_detector = vision.HandLandmarker.create_from_options(options)
connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

finger_tips = (4, 8, 12, 16, 20)
angle_threshold = 160
finger_angle_points = (
    (1, 2, 3),
    (5, 6, 7),
    (9, 10, 11),
    (13, 14, 15),
    (17, 18, 19),
)

# 모델 선언
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
if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다.")

previous_box = None
crop_window_open = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라 프레임을 읽지 못했습니다.")
        break

    # 카메라 설치 방향에 맞춘 상하 반전. 좌우 반전이 필요하면 0을 1로 변경한다.
    frame = cv2.flip(frame, 0)
    frame_height, frame_width = frame.shape[:2]

    # MediaPipe 손 탐지
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    hand_result = hand_detector.detect(mp_image)
    labels = [
        "Left" if handedness[0].category_name == "Right" else "Right"
        for handedness in hand_result.handedness
    ]

    total_finger_count = 0
    for hand in hand_result.hand_landmarks:
        angles = [
            calculate_angle(hand[p1], hand[p2], hand[p3])
            for p1, p2, p3 in finger_angle_points
        ]
        total_finger_count += sum(angle >= angle_threshold for angle in angles)

        if (
            angles[0] >= angle_threshold
            and angles[1] >= angle_threshold
            and all(angle < angle_threshold for angle in angles[2:])
        ):
            print("OKAY")

    cv2.putText(
        frame,
        f"Hands: {len(hand_result.hand_landmarks)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Fingers: {total_finger_count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    handedness_text = " / ".join(labels)
    text_size = cv2.getTextSize(
        handedness_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    )[0]
    text_x = frame_width - text_size[0] - 20
    cv2.putText(
        frame,
        handedness_text,
        (text_x, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    for hand in hand_result.hand_landmarks:
        points = [
            (int(point.x * frame_width), int(point.y * frame_height))
            for point in hand
        ]
        for connection in connections:
            cv2.line(
                frame,
                points[connection.start],
                points[connection.end],
                (0, 255, 0),
                2,
            )
        for index, point in enumerate(points):
            color = (0, 0, 255) if index in finger_tips else (255, 0, 0)
            radius = 6 if index in finger_tips else 4
            cv2.circle(frame, point, radius, color, -1)

    # YOLO 신호등(class 9) 탐지
    yolo_result = yolo.predict(
        source=frame,
        conf=YOLO_CONFIDENCE,
        iou=YOLO_IOU,
        verbose=False,
        classes=[9],
    )[0]

    selected = select_traffic_light(
        yolo_result.boxes, frame_width, frame_height
    )

    if selected is not None:
        current_box, confidence = selected
        previous_box = smooth_box(previous_box, current_box)
        x1, y1, x2, y2 = clamp_and_pad_box(
            previous_box, frame_width, frame_height
        )

        if x2 > x1 and y2 > y1:
            traffic_light_image = frame[y1:y2, x1:x2].copy()
            if traffic_light_image.size > 0:
                display_image = cv2.resize(
                    traffic_light_image,
                    (DISPLAY_WIDTH, DISPLAY_HEIGHT),
                    interpolation=cv2.INTER_NEAREST,
                )
                cv2.imshow("Traffic Light", display_image)
                crop_window_open = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"traffic light {confidence:.2f}"
            label_y = max(25, y1 - 10)
            cv2.putText(
                frame,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )
    else:
        previous_box = None
        cv2.putText(
            frame,
            "Traffic light: not detected",
            (20, frame_height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        if crop_window_open:
            cv2.destroyWindow("Traffic Light")
            crop_window_open = False

    cv2.imshow("YOLO Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
hand_detector.close()
cv2.destroyAllWindows()
