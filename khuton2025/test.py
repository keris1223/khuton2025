import pandas as pd

# CSV 파일 불러오기
csv_path = "name_temp_hum_light.csv"  # 사용자의 파일 경로
plant_df = pd.read_csv(csv_path)

# 문자열로 저장된 튜플을 실제 튜플로 변환
plant_df["temperature"] = plant_df["temperature"].apply(eval)
plant_df["humidity"] = plant_df["humidity"].apply(eval)
plant_df["light"] = plant_df["light"].apply(eval)

# 조건을 모두 만족하는 경우에만 점수를 계산하는 추천 함수
def strict_score_based_recommendation_by_light(temp, hum, lights, df, max_recommend=4):
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
            temp_score = max(0, 1.5 - (temp_diff / 10))
            t_mm = abs(t_max - t_min)
            temp_tmmscore = max(0, 1.5 - (t_mm/5))
            score += (0.4 * temp_score) + (0.6 * temp_tmmscore)

            # 2. 습도 점수 (편차 작을수록 가산)
            h_opt = (h_min + h_max) / 2
            hum_diff = abs(h_opt - hum)
            hum_score = max(0, 1.5 - (hum_diff / 40))
            h_mm = abs(h_max - h_min)
            hum_hmmscore = max(0, 1.5 - (h_mm/10))
            score += (0.4 * hum_score) + (0.6 * hum_hmmscore)

            # 3. 조도 점수 (편차 작을수록 가산)
            l_opt = (l_min + l_max) / 2
            lux_diff = abs(l_opt - light)
            light_score = max(0, 1.5 - (lux_diff / 30000))
            l_mm = abs(l_max - l_min)
            hux_lmmscore = max(0, 1.5- (l_mm/10000))
            score += (0.4 * light_score) + (0.6 * hux_lmmscore)

            scored_plants.append({
                "조도": light,
                "식물": name,
                "점수": round(score, 4)
            })

        # 점수 내림차순 정렬 후 상위 N개만 저장
        top_plants = sorted(scored_plants, key=lambda x: x["점수"], reverse=True)[:max_recommend]
        results[light] = top_plants

    # 전체 결과 하나의 리스트로 결합
    all_results = []
    for plant_list in results.values():
        all_results.extend(plant_list)

    return pd.DataFrame(all_results)

# 예시 센서 값
sensor_temperature = 22
sensor_humidity = 65
sensor_lights = [10000, 25000, 50000]

# 추천 실행
df_recommend = strict_score_based_recommendation_by_light(sensor_temperature, sensor_humidity, sensor_lights, plant_df)

# 결과 출력
print("🌱 조도별 상위 4개 추천 식물:")
print(df_recommend)
