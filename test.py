import pandas as pd
p = "/mnt/data/features/tempo/nrt_roll3d/nrt_merged.parquet"
df = pd.read_parquet(p)

print(df.dtypes)                        # 최소: time, lat, lon, no2
print(df.columns.tolist())             # o3/기타 컬럼 존재 여부
print(df['time'].min(), df['time'].max(), df['time'].nunique())
print(df[['lat','lon']].drop_duplicates().shape)  # 격자 크기
print(df['no2'].isna().mean(), df['no2'].describe())  # 결측/스케일
print(df.sort_values('time').groupby(['lat','lon']).time.diff().describe())  # 시간 간격
