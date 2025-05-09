from datetime import datetime
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

# 한글 폰트 설정
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# CSV 파일 불러오기
csv_path = "name_temp_hum_light.csv"  # 사용자의 파일 경로
plant_df = pd.read_csv(csv_path)

# 문자열로 저장된 튜플을 실제 튜플로 변환
plant_df["temperature"] = plant_df["temperature"].apply(eval)
plant_df["humidity"] = plant_df["humidity"].apply(eval)
plant_df["light"] = plant_df["light"].apply(eval)

# 조건을 모두 만족하는 경우에만 점수를 계산하는 추천 함수
def strict_score_based_recommendation_by_light(temp, hum, lights, df, max_recommend=1):
    results = {}

    for light in lights:
        scored_plants = []

        for _, row in df.iterrows():
            name = row["name"]
            t_min, t_max = row["temperature"]
            h_min, h_max = row["humidity"]
            l_min, l_max = row["light"]

            # ✅ 모든 기준을 만족해야만 점수 계산 시작
            if not (t_min <= temp <= t_max):
                continue
            if not (h_min <= hum <= h_max):
                continue
            if not (l_min <= light <= l_max):
                continue

            score = 0

            # 1. 온도 적합 → 기본 점수
            t_opt = (t_min + t_max) / 2
            temp_diff = abs(t_opt - temp)
            temp_score = max(0, 1 - (temp_diff / 13))
            t_mm = abs(t_max - t_min)
            temp_tmmscore = max(0, 1 - (t_mm/13))
            score += (0.4 * temp_score) + (0.6 * temp_tmmscore)

            # 2. 습도 점수 (편차 작을수록 가산)
            h_opt = (h_min + h_max) / 2
            hum_diff = abs(h_opt - hum)
            hum_score = max(0, 1 - (hum_diff / 50))
            h_mm = abs(h_max - h_min)
            hum_hmmscore = max(0, 1 - (h_mm/50))
            score += (0.4 * hum_score) + (0.6 * hum_hmmscore)

            # 3. 조도 점수 (편차 작을수록 가산)
            l_opt = (l_min + l_max) / 2
            lux_diff = abs(l_opt - light)
            light_score = max(0, 1 - (lux_diff / 50000))
            l_mm = abs(l_max - l_min)
            hux_lmmscore = max(0, 1- (l_mm/50000))
            score += (0.4 * light_score) + (0.6 * hux_lmmscore)

            scored_plants.append({
                "조도": light,
                "식물": name,
                "점수": round(score, 4)
            })

        # 점수 내림차순 정렬 후 상위 N개만 저장
        if scored_plants:
            top_plants = sorted(scored_plants, key=lambda x: x["점수"], reverse=True)[:max_recommend]
        else:
            top_plants = [{
                "조도": light,
                "식물": "너무 밝음",
                "점수": "0"
            }]
        results[light] = top_plants

    # 전체 결과 하나의 리스트로 결합
    all_results = []
    for plant_list in results.values():
        all_results.extend(plant_list)

    return pd.DataFrame(all_results)

# 예시 센서 값

#sensor_temperature = 22
#sensor_humidity = 65
#sensor_lights = [10000, 25000, 50000]

# 월별 평균 기온/습도 파일 불러오기
monthly_env_path = "월별_평균_기온_습도(수원).csv"
monthly_env_df = pd.read_csv(monthly_env_path)

# 현재 시간에 맞는 달 가져오기
current_month = datetime.now().month
month_str = f"{current_month}월"

# 현재 월의 평균 기온/습도 가져오기
row = monthly_env_df[monthly_env_df["월"] == month_str].iloc[0]
sensor_temperature = round(row["평균기온(℃)"],4)
sensor_humidity = round(row["평균습도(%)"],4)

# LUX 값 가져오기
df = pd.read_csv("brightness_array.csv", header=None)
brightness = df.values
print(f"원본 밝기 배열 크기: {brightness.shape}")


flat = brightness.reshape(-1, 1)  
cluster_num = int(input("작물의 종류가 몇개인가요:")) # 심을 작물 개수
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

sensor_lights = lux_list

print(f"📅 현재 월: {month_str}")
print(f"🌡️ 평균기온: {sensor_temperature} ℃")
print(f"💧 평균습도: {sensor_humidity} %")

# 추천 실행
df_recommend = strict_score_based_recommendation_by_light(sensor_temperature, sensor_humidity, sensor_lights, plant_df)

# 결과 출력
print("🌱 조도별 추천 식물:")
print(df_recommend)

# 시각화
cmap = plt.cm.get_cmap('viridis', cluster_num)
colors = [cmap(i) for i in range(cluster_num)]
recommend_map = dict(zip(df_recommend["조도"], df_recommend["식물"]))

plt.figure(figsize=(12, 6))
plt.imshow(segmented, cmap=cmap)
plt.title("클러스터별 추천 작물 시각화")
plt.axis('off')

min_region_size = 6500
max_lux_exclude = 80000

legend_patches_name = []

for cluster_id in range(cluster_num):
    cluster_lux = math.floor((centers[cluster_id] / 255.0) * 100000)
    crop_name = recommend_map.get(cluster_lux, "너무 밝음")
    legend_patches_name.append(mpatches.Patch(color=colors[cluster_id], label=crop_name))

    cluster_mask = (segmented == cluster_id).astype(int)
    labeled, _ = label(cluster_mask)
    slices = find_objects(labeled)

    for i, slc in enumerate(slices):
        component = (labeled[slc] == (i + 1)).astype(int)
        if np.sum(component) < min_region_size:
            continue
        y_idx, x_idx = np.where(component)
        center_y = y_idx.mean() + slc[0].start
        center_x = x_idx.mean() + slc[1].start
        plt.text(int(center_x), int(center_y), crop_name, fontsize=10, color='white',
                 ha='center', va='center', bbox=dict(facecolor='red' if crop_name == "너무 밝음" else 'black',
                                                     alpha=0.6, boxstyle='round'))

# Remove duplicate legend entries
unique_legend = {patch.get_label(): patch for patch in legend_patches_name}
plt.legend(handles=list(unique_legend.values()), loc='upper left', fontsize=9, frameon=True)
plt.tight_layout()
plt.show()