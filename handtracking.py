import cv2
import mediapipe as mp
import numpy as np
import time
import math

# 1. Khởi tạo MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# danh sách danh mục bộ lọc màu
FILTER_NAMES = [
    "1. CYBERPUNK NEON (Pink/Cyan)",
    "2. THERMAL HEATMAP (Nhiet)",
    "3. MATRIX GREEN (Hacker)",
    "4. VINTAGE SEPIA (Co dien)",
    "5. NEGATIVE INVERT (Am ban)",
    "6. MONO CHIC (Den trang)"
]
current_filter_idx = 0

# Hàm áp dụng các bộ lọc màu cực chất
def apply_style_filter(roi, mode_idx):
    if mode_idx == 0:  # Cyberpunk Neon
        # Tăng sắc Hồng / Xanh Neon
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0] + 130) % 180  # Shift màu sang hồng/tím
        hsv[:, :, 1] = cv2.add(hsv[:, :, 1], 80)    # Tăng độ rực màu (Saturation)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    elif mode_idx == 1:  # Thermal / Heatmap
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    elif mode_idx == 2:  # Matrix Green
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        matrix = np.zeros_like(roi)
        matrix[:, :, 1] = cv2.add(gray, 50)  # Chỉ giữ kênh màu Green
        return matrix

    elif mode_idx == 3:  # Vintage Sepia
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        return cv2.transform(roi, kernel)

    elif mode_idx == 4:  # Negative Invert
        return cv2.bitwise_not(roi)

    elif mode_idx == 5:  # Mono Chic (Đen trắng tăng tương phản)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return cv2.convertScaleAbs(gray_3ch, alpha=1.3, beta=5)

    return roi

# Hàm vẽ góc Cyberpunk Neon
def draw_cyber_corners(img, pt1, pt2, color=(0, 255, 255), thickness=2, length=20):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1, y1), (x2, y2), (60, 60, 60), 1)
    
    # 4 góc nổi bật
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)

# Hàm vẽ HUD Overlay
def draw_hud_overlay(img, filter_name):
    h, w, _ = img.shape
    
    # REC Chớp tắt
    if int(time.time() * 2) % 2 == 0:
        cv2.circle(img, (40, 40), 8, (0, 0, 255), -1)
    cv2.putText(img, "REC", (55, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Timestamp
    date_str = time.strftime("%d.%m.%Y %H:%M:%S")
    cv2.putText(img, date_str, (w - 240, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Tên Filter Đang Chọn
    cv2.putText(img, f"FILTER: {filter_name}", (40, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(img, "[Cham 2 ngon tro / Bam 'C' de doi mau]", (w - 380, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

last_switch_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    points = []
    index_tips = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger = hand_landmarks.landmark[8]
            thumb_finger = hand_landmarks.landmark[4]

            cx_i, cy_i = int(index_finger.x * w), int(index_finger.y * h)
            cx_t, cy_t = int(thumb_finger.x * w), int(thumb_finger.y * h)

            points.append((cx_i, cy_i))
            points.append((cx_t, cy_t))
            index_tips.append((cx_i, cy_i))

            # Draw fingertip neon circles
            cv2.circle(frame, (cx_i, cy_i), 6, (0, 255, 255), -1)

    # CỬ CHỈ ĐỔI BỘ LỌC MÀU: Chạm 2 ngón trỏ lại gần nhau
    if len(index_tips) == 2:
        dist = math.hypot(index_tips[0][0] - index_tips[1][0], index_tips[0][1] - index_tips[1][1])
        if dist < 35 and (time.time() - last_switch_time) > 1.2:  # Cooldown 1.2 giây
            current_filter_idx = (current_filter_idx + 1) % len(FILTER_NAMES)
            last_switch_time = time.time()

    # RENDER BỘ LỌC NỀN TRONG KHUNG TAY
    if len(points) >= 4:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        x1, y1 = max(0, min(xs)), max(0, min(ys))
        x2, y2 = min(w, max(xs)), min(h, max(ys))

        if (x2 - x1) > 60 and (y2 - y1) > 60:
            roi = frame[y1:y2, x1:x2]
            
            # Áp dụng màu theo Filter đang được chọn
            styled_roi = apply_style_filter(roi, current_filter_idx)
            frame[y1:y2, x1:x2] = styled_roi
            
            draw_cyber_corners(frame, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=2)

    draw_hud_overlay(frame, FILTER_NAMES[current_filter_idx])

    cv2.imshow("RETROLENS - Multi-Color Hand Tracking", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):  # Đổi màu thủ công bằng phím C
        current_filter_idx = (current_filter_idx + 1) % len(FILTER_NAMES)

cap.release()
cv2.destroyAllWindows()