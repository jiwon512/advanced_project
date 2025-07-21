from hye_project.my_package.stat_file import normality, stat_test, outlier

# -------
# library
# -------

# Standard library
from itertools import combinations

# Typing
from typing import Any, Dict, Tuple, Union

# Data handling
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# Reporting
from tabulate import tabulate

# Statistical analysis
import scipy.stats as stats
from scipy.stats import levene
from scipy.spatial.distance import squareform

# Clustering
import scipy.cluster.hierarchy as sch

# Post-hoc tests
import scikit_posthocs as sp

# Modeling
import statsmodels.api as sm

# Network analysis
import networkx as nx

# ------------------------------
# 전처리를 완료한 csv 파일 불러오기
# ------------------------------
df = pd.read_csv('/Users/hyeom/Documents/GitHub/advanced_project/jiwon_project/csv_files/preprocessing_filtered.csv', index_col=0)
'''

# ------------------------------
# price 컬럼 정규성 및 분포 확인하기
# ------------------------------
# 1. price 컬럼 정규성 검정 결과 / QQ plot 시각화
print('=== price 정규성 검정 ===')
price_normality = normality.test(df['price']);

plt.figure(figsize=(8, 6))
stats.probplot(df['price'], dist="norm", plot=plt)
plt.title("Price QQ-Plot")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.tight_layout()
plt.show()
'''

# 2. price가 이상치로 인해 정규성을 따르지 않기 때문에, log_price 컬럼 추가
df['log_price'] = np.log1p(df['price'])
'''

print('=== log price 정규성 검정 ===')
log_price_normality = normality.test(df['log_price']);

plt.figure(figsize=(8, 6))
stats.probplot(df['log_price'], dist="norm", plot=plt)
plt.title("Log Price QQ-Plot")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.tight_layout()
plt.show()

print('\n[이상치 제거 전 price describe]')
print(df['price'].describe())

# -> log_price도 정규성을 띄진 않지만, 왜도 0.57, 첨도 1.04로 정규성을 띄고 있다고 가정
# -> log_price를 기준으로 이상치 제거 기준을 가설 검정을 통해 찾고, 이상치를 제거

# ------------------------------
# 이상치를 정의할 기준 컬럼 찾기
# 1) room_type과 price의 연관성
# ------------------------------
X1 = 'room_type'
y = 'log_price'
alpha = 0.05
max_shapiro_n = 5000

print('\n=== room type 과 price 연관성 가설 검정 ===')
# 1. 정규성과 등분산성 검정 후, Kruskal-Wallis 검정 진행
X1_stat_test = stat_test.decide(df, X1, y, alpha=alpha, verbose=True);

# 2. Kruskal-Wallis 검정 진행, 사후 검정은 Dunn(holm 보정)
X1_res = stat_test.kruskal_dunn(df, X1, y, alpha=alpha, adjust='holm', verbose=True);

# 3. 검정결과 시각화
if X1_res['pvals_matrix'] is not None:
    stat_test.p_heatmap(
        X1_res['pvals_matrix'],
        alpha=alpha,
        clip_upper=0.05,
        annot_mode="p",
        cmap="rocket_r",
        figsize=(8, 6),
        text_color="black",
    )
    plt.show()

# 4. room type으로 이상치 시각화
outlier.boxplot(df, X1, y, factor=1.5, figsize=(8,6), tablefmt='github', verbose=True)

# -> 적어도 한 개 이상의 room type 쌍의 price 분포가 통계적으로 다르다.
# -> 따라서 room type 을 사용하여 price 이상치를 판단할 수는 있지만,
# -> 박스플롯 확인 결과 room type 은 price 이상치를 완전하게 설명하지 못한다.
'''
# ------------------------------
# 이상치를 정의할 기준 컬럼 찾기
# 2) room_structure_type price의 연관성
# ------------------------------
X2 = 'room_structure_type'
y = 'log_price'
alpha = 0.05
max_shapiro_n = 5000

print('\n=== room structure type 과 price 연관성 가설 검정 ===')
# 1. 정규성과 등분산성 검정 후, Kruskal-Wallis 검정 진행
X2_stat_test = stat_test.decide(df, X2, y, alpha=alpha, verbose=True);

# 2. Kruskal-Wallis 검정 진행, 사후 검정은 Dunn(holm 보정)
X2_res = stat_test.kruskal_dunn(df, X2, y, alpha=alpha, adjust='holm', verbose=True);

# 3. 검정결과 시각화
if X2_res['pvals_matrix'] is not None:
    stat_test.p_heatmap(
        X2_res['pvals_matrix'],
        alpha=alpha,
        clip_upper=0.05,
        annot_mode="none",
        cmap="rocket_r",
        figsize=(8, 6),
        text_color="black",
    )
    plt.show()

# 4. room structure type 으로 이상치 시각화
outlier.boxplot(df, X2, y, factor=1.5, figsize=(8,6), tablefmt='github', verbose=True);

# -> 적어도 한 개 이상의 room structure type 쌍의 price 분포가 통계적으로 다르다.
# -> 따라서 room structure type 을 사용하여 price 이상치를 판단할 수는 있지만,
# -> 박스플롯 확인 결과 room structure type 은 price 이상치를 완전하게 설명하지 못한다.

# ------------------------------------------------------------------------------------------
# 이상치를 정의할 기준 컬럼 찾기
# 3) room_structure_type 그룹 간 p-value 검사 결과를 바탕으로 새로운 카테고리 제안 (p-value 거리로 군집화)
# ------------------------------------------------------------------------------------------
pmat = X2_res['pvals_matrix']          # Dunn 사후 p-value DataFrame
index = pmat.index

# 1. p 값을 [ε, 1] 범위로 고정
P = np.clip(pmat.values, 1e-10, 1.0)

# 2. 거리 = -log10(p),  p=1 → 0
D = -np.log10(P)
np.fill_diagonal(D, 0)

# 3. linkage (average·k=5 예시)
Z      = sch.linkage(squareform(D), method='average')
labels = sch.fcluster(Z, t=5, criterion='maxclust')

# 4. 매핑
struct_grp_map = dict(zip(index, labels))
df['room_new_type'] = df['room_structure_type'].map(struct_grp_map)

print("\n=== 군집화로 도출한 새로운 그룹 카테고리 ===")
for k in sorted(set(labels)):
    print(f"Group {k}: {[s for s,l in struct_grp_map.items() if l==k]}")

# 5. 군집별 표본 수 & 로그 가격 통계
grp_stat = (
    df.groupby('room_new_type')['price']
      .agg(n='size', median='median', q1=lambda s: s.quantile(.25), q3=lambda s: s.quantile(.75))
      .sort_values('median')
)

# 6. 중앙값과 2사분위, 3사분위를 고려하여 재배치
# - 5번그룹(townhouse)은 금액 특성상 4번그룹에 통합 가능
# - barn, kezhan, ranch, dome은 개수가 부족해 군집화가 불가능 -> 적합한 가격군에 배치
df.loc[df.room_structure_type == 'townhouse', 'room_new_type'] = 3
df.loc[df.room_structure_type == 'barn', 'room_new_type'] = 1
df.loc[df.room_structure_type == 'kezhan', 'room_new_type'] = 4
df.loc[df.room_structure_type == 'ranch', 'room_new_type'] = 4
df.loc[df.room_structure_type == 'dome', 'room_new_type'] = 4

# 7. 그룹명 변경
# - 금액대 유사한 그룹끼리 배치했기에, high, upper mid, mid, low mid로 변경
df['room_new_type'] = df['room_new_type'].astype(int)

group_name_map = {4: "Low-Mid", 3: "Mid", 1: "Upper-Mid",  2: "High"}
df['room_new_type'] = df['room_new_type'].map(group_name_map)

# 8. room new type 과 price 연관성 - kruskal wallis 검정
new_res = stat_test.kruskal_dunn(df, X='room_new_type', y='price', alpha=alpha, adjust='holm', verbose=True);

#9. 박스플롯
bounds = outlier.robust_bounds(df, 'room_new_type', 'price', k=3)

df = df.merge(bounds, on='room_new_type', how='left')
df['is_outlier'] = (df['price'] < df['lower']) | (df['price'] > df['upper'])

plot_df = df.copy()

# Separate clean data and outliers
clean_df = plot_df[~plot_df['is_outlier']]
out_df   = plot_df[plot_df['is_outlier']]

fig, ax = plt.subplots(figsize=(10, 6))

# Box‑and‑whisker for each structure_group (clean data only)
groups = sorted(clean_df['room_new_type'].unique())
data   = [clean_df.loc[clean_df['room_new_type'] == g, 'price'] for g in groups]

ax.boxplot(data, labels=groups, showfliers=True)  # hide internal fliers

ax.set_xlabel("room_new_type")
ax.set_ylabel("Price ($)")
ax.set_title("Price distribution by room_new_type Group (outliers removed)")

plt.tight_layout()
plt.show()

# -> kruskal 검정 결과, 유의하지 않은 그룹쌍은 없었다.
# -> 이후, 박스플롯으로 이상치를 확인해보면 해당 그룹은 이상치를 잘 설명하고 있음을 알 수 있다.

stats_type = df.groupby("room_new_type")['price'].apply(outlier.stats)
print("=== room_new_type 별 price 이상치 ===")

# 인덱스를 컬럼으로 올리기
stats_type = stats_type.reset_index()

# apply 결과를 MultiIndex → DataFrame 으로 펼치기
stats_type = (
    df.groupby("room_new_type")['price']
      .apply(outlier.stats)
      .unstack()              # outlier_count, outlier_ratio 가 각각 컬럼이 됨
      .reset_index()          # 구조를 DataFrame으로 완성
)
type_outlier_count = stats_type['outlier_count'].sum()
type_outlier_ratio = stats_type['outlier_count'].sum() / 22308

print(f"전체 이상치 개수: {type_outlier_count}")
print(f"전체 이상치 비율: {type_outlier_ratio:.4f}")

df.groupby('room_new_type')['price'].describe()
stats_clean = (
    df.groupby('room_new_type')['price']
      .apply(outlier.describe_without)   # → 다중 인덱스 Series
      .unstack()                    # → 행: structure_group, 열: describe 항목
)

mask = df.groupby('room_new_type')['price'] \
         .transform(lambda s: outlier.is_not(s, factor=1.5))

# 2) 이상치가 아닌 행만 골라 새로운 DataFrame에 저장
outlier_removed_df = df[mask].copy()

# 3) 확인 (원본 vs 제거 후 행 개수)
print(f"Original rows: {len(df)}, Without outliers: {len(outlier_removed_df)}")

# 4) 필요하다면 인덱스 리셋
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

df = df.loc[~df['is_outlier']].reset_index(drop=True)

df = df.drop(columns=['median', 'lower', 'upper', 'n', 'is_outlier'])
df['room_new_type'] = df['room_new_type'].str.lower()

# 3) 인덱스를 0,1,2… 로 재설정하고 파일에 저장
df = df.reset_index(drop=True)
df.index.name = ''
df.to_csv(
    '/Users/hyeom/Documents/GitHub/advanced_project/Airbnb_project_15/outlier_removed.csv',
    index=False,
    header=True,
    encoding='utf-8'
)
print("Saved without 'Unnamed: 0' and without index.")