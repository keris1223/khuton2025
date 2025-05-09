import os
os.environ["OMP_NUM_THREADS"] = "2" 

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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

list = []

for i, (b, l) in enumerate(sorted_pairs):
    print(f"클러스터 {i}: 밝기 {b:.2f}, 약 {l:,.0f} lux")
    list.append(math.floor(l))

    strlis = []

for i in range(cluster_num):
    if list[i] <= 10000:
        strlis.append("상추")
    elif list[i] <= 20000:
        strlis.append("브로콜리")
    elif list[i] <= 30000:
        strlis.append("고추")
    elif list[i] <= 40000:
        strlis.append("오이")
    elif list[i] <= 50000:
        strlis.append("토마토")
    elif list[i] <= 60000:
        strlis.append("애호박")
    elif list[i] <= 70000:
        strlis.append("수박")
    elif list[i] <= 80000:
        strlis.append("멜론")
    elif list[i] <= 90000:
        strlis.append("해바라기")
    elif list[i] <= 100000:
        strlis.append("너무 밝음")

for i in range(cluster_num):
    print(strlis[i])