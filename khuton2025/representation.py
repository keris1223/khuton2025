import os
os.environ["OMP_NUM_THREADS"] = "2" 

from scipy.ndimage import label, find_objects
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

df = pd.read_csv("output/brightness_array.csv", header=None)
brightness = df.values
print(f"원본 밝기 배열 크기: {brightness.shape}")


flat = brightness.reshape(-1, 1)  
cluster_num = int(input("심을 작물의 가짓수를 입력해 주세요 :")) # 심을 작물 개수
kmeans = KMeans(n_clusters=cluster_num, random_state=42)
kmeans.fit(flat)
labels = kmeans.labels_

segmented = labels.reshape(brightness.shape)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(brightness, cmap='gray')
plt.title('Original Brightness Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented, cmap='viridis')
plt.title(f"{cluster_num}-Region K-Means Segmentation")
plt.axis('off')

plt.tight_layout()
plt.show()

centers = kmeans.cluster_centers_.flatten()
sorted_centers = sorted(centers)
for i, c in enumerate(sorted_centers):
    print(f"클러스터 {i} 중심 밝기: {c:.2f}")

centers = kmeans.cluster_centers_.flatten()

max_lux = 100000

lux_values = (centers / 255.0) * max_lux

sorted_pairs = sorted(zip(centers, lux_values))  

print("클러스터 중심 밝기 및 Lux 변환값:")

lux_list = []

for i, (b, l) in enumerate(sorted_pairs):
    print(f"클러스터 {i}: 밝기 {b:.2f}, 약 {l:,.0f} lux")
    lux_list.append(math.floor(l))
    
    strlis = []

light_thresholds = {
    10000: "상추",
    20000: "브로콜리",
    30000: "고추",
    40000: "오이",
    50000: "토마토",
    60000: "애호박",
    70000: "수박",
    80000: "멜론",
    90000: "해바라기",
    100000: "너무 밝음"
}



for value in lux_list[:cluster_num]:  # lux_list의 앞 3개 항목만 사용
    for threshold, crop in light_thresholds.items():
        if value <= threshold:
            strlis.append(crop)
            break

for i in range(cluster_num):
    print(strlis[i])

cmap = plt.cm.get_cmap('viridis', cluster_num)
colors = [cmap(i) for i in range(cluster_num)]

# 색상-작물 매핑용 패치 리스트 생성
legend_patches = [
    mpatches.Patch(color=colors[i], label=strlis[i]) for i in range(cluster_num)
]

plt.figure(figsize=(10, 5))
plt.imshow(segmented, cmap='viridis')
plt.title(f"{cluster_num}-Region K-Means Segmentation")
plt.axis('off')

min_region_size = 6500  # 작은 영역 필터링 기준 (픽셀 수)

for cluster_id in range(cluster_num):
    # 현재 클러스터에 해당하는 마스크
    cluster_mask = (segmented == cluster_id).astype(int)

    # 연결된 구성요소 레이블링
    labeled, num_features = label(cluster_mask)
    slices = find_objects(labeled)

    for i, slc in enumerate(slices):
        # 해당 덩어리 좌표 추출
        component = (labeled[slc] == (i + 1)).astype(int)
        area = np.sum(component)

        if area < min_region_size:
            continue  # 너무 작으면 생략

        # 중심 위치 계산
        y_indices, x_indices = np.where(component)
        center_y = y_indices.mean() + slc[0].start
        center_x = x_indices.mean() + slc[1].start

        # 텍스트 표시
        label_name = strlis[cluster_id]
        plt.text(int(center_x), int(center_y), label_name, fontsize=10, color='white',
                 ha='center', va='center', bbox=dict(facecolor='black', alpha=0.6, boxstyle='round'))

# 범례 추가
plt.legend(handles=legend_patches, loc='upper left', fontsize=9, frameon=True)

plt.tight_layout()
plt.show()