import cv2
import numpy as np

# 사용자 설정
frame_interval = 30
accumulated_frames = 100
plant_w, plant_h = 50, 50  # 화분 크기 (픽셀)

def find_max_brightness_area(brightness_map, w, h):
    max_val = -1
    max_pos = (0, 0)

    for y in range(0, brightness_map.shape[0] - h):
        for x in range(0, brightness_map.shape[1] - w):
            region = brightness_map[y:y+h, x:x+w]
            avg = np.mean(region)

            if avg > max_val:
                max_val = avg
                max_pos = (x, y)

    return max_pos, max_val  # 좌상단 위치 + 평균 밝기

def brightness_to_lux(avg_brightness):
    # 밝기(0~255)를 기준으로 lux 근사 (255 = 20000 lux 기준)
    return (avg_brightness / 255.0) * 20000

cap = cv2.VideoCapture(0)
frame_count = 0
brightness_sum = None

cv2.namedWindow("Live Feed")
cv2.namedWindow("Heatmap + Recommendation")

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
        normalized = cv2.convertScaleAbs(avg_brightness_map)  # 밝기값을 0~255로 클리핑 없이 직접 정수화


        # 가장 밝은 영역 찾기
        (x, y), max_region_avg = find_max_brightness_area(normalized, plant_w, plant_h)
        est_lux = brightness_to_lux(max_region_avg)

        # 히트맵 생성
        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_HOT)
        heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

        # 위치 스케일 보정
        scale_x = frame.shape[1] / normalized.shape[1]
        scale_y = frame.shape[0] / normalized.shape[0]
        pt1 = (int(x * scale_x), int(y * scale_y))
        pt2 = (int((x + plant_w) * scale_x), int((y + plant_h) * scale_y))

        # 추천 영역 표시 + lux 표시
        cv2.rectangle(heatmap_resized, pt1, pt2, (255, 255, 255), 2)
        cv2.putText(heatmap_resized, f"Best Spot: ~{int(est_lux)} lux",
                    (pt1[0]+5, pt1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Live Feed에도 동일 표시
        cv2.rectangle(display_frame, pt1, pt2, (0, 255, 255), 2)
        cv2.putText(display_frame, f"{int(est_lux)} lux",
                    (pt1[0]+5, pt1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        cv2.imshow("Heatmap + Recommendation", heatmap_resized)

    # Live Feed 항상 출력
    cv2.imshow("Live Feed", display_frame)

    key = cv2.waitKey(1)
    if key == 27 or (frame_count // frame_interval) >= accumulated_frames:
        break

cap.release()
cv2.destroyAllWindows()
