import cv2
import numpy as np

# 사용자 설정
frame_interval = 30
accumulated_frames = 100
plant_w, plant_h = 50, 50  # 화분 크기 (픽셀)


cap = cv2.VideoCapture(0)
frame_count = 0
brightness_sum = None

cv2.namedWindow("Live Feed")
cv2.namedWindow("Heatmap")

print("[INFO] 실시간 분석 시작. ESC로 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    display_frame = frame.copy()

    if frame_count % frame_interval == 0:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]

        if brightness_sum is None:
            brightness_sum = np.zeros_like(v, dtype=np.float32)

        brightness_sum += v.astype(np.float32)

        avg_brightness_map = brightness_sum / (frame_count // frame_interval)
        # 밝기 기준값 설정 (예: 200 이상이면 최고 광량으로 간주)
        brightness_max_reference = 255

        # 1. 기준값 기준으로 정규화하고
        normalized = (avg_brightness_map / brightness_max_reference) * 255

        # 2. 범위 클리핑 (255 초과 방지)
        normalized = np.clip(normalized, 0, 255)

        # 3. 정수 타입으로 변환
        normalized = normalized.astype(np.uint8)

        # 히트맵 생성
        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_HOT)
        heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        np.savetxt("brightness_array.csv", avg_brightness_map, delimiter=",")

        cv2.imshow("Heatmap", heatmap_resized)
        cv2.imwrite("heatmap.png", heatmap)

    # Live Feed 항상 출력
    cv2.imshow("Live Feed", display_frame)
    cv2.imwrite("live.png",display_frame)
    key = cv2.waitKey(1)
    if key == 27 or (frame_count // frame_interval) >= accumulated_frames:
        break

cap.release()
cv2.destroyAllWindows()
