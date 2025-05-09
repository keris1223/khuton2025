import pandas as pd

df_raw = pd.read_csv("suwonweather.csv", header=[0, 1, 2], encoding="utf-8")

df_raw = df_raw.set_index(("월별(1)", "월별(1)", "월별(1)"))

df_raw = df_raw.apply(pd.to_numeric, errors='coerce')

temp_cols = [col for col in df_raw.columns if "기온" in col[1]]
humidity_cols = [col for col in df_raw.columns if "상대습도" in col[1]]

df_monthly_avg = pd.DataFrame({
    "평균기온(℃)": df_raw[temp_cols].mean(axis=1),
    "평균습도(%)": df_raw[humidity_cols].mean(axis=1)
})

df_monthly_avg = df_monthly_avg.drop("연간", errors="ignore")

df_monthly_avg.index.name = "월"
df_monthly_avg.index = df_monthly_avg.index.astype(str)


print(df_monthly_avg.round(1))

df_monthly_avg.to_csv("월별_평균_기온_습도(수원).csv", encoding='utf-8-sig')
