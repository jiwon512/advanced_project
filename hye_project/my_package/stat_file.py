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


# -----------
# My function!
# -----------

# 정규성 검정 결과
def test_normality(
    series: pd.Series,
    alpha: float = 0.05,
    verbose: bool = True
) -> Dict[str, Any]:

    """
    주어진 수치형 시리즈에 대해 여러 정규성 검정을 수행하고 결과를 딕셔너리로 반환합니다.

    Args:
        series: pandas Series, numeric 데이터 (NaN은 자동 제거)
        alpha: 유의수준 (p-value 비교 기준)
        verbose: True일 경우 결과 요약을 콘솔에 출력

    Returns:
        Dict[str, Any]:
            {
                'n': 표본 수,
                'skew': 왜도 (정규분포에서 0),
                'kurt_excess': 초과 첨도 (정규분포에서 0),

                'shapiro_stat': Shapiro-Wilk 통계량,
                'shapiro_p' : Shapiro-Wilk p-value,
                'dagostino_k2': D’Agostino K² 통계량,
                'dagostino_p': D’Agostino K² p-value,
                'jarque_bera': Jarque–Bera 통계량,
                'jarque_bera_p': Jarque–Bera p-value,
                'anderson_stat': Anderson–Darling 통계량,
                'anderson_crit': {유의수준: 임계값, ...},
                'lilliefors_stat': Lilliefors KS 통계량,
                'lilliefors_p': Lilliefors p-value,

                'practical_normal': 실무적 판정 (skew<0.5 & |kurt|<1)
            }
    """

    # 1. 데이터 전처리
    arr = pd.to_numeric(series, errors='coerce').dropna().values
    n = arr.size

    # 2. 기초 통계량
    skewness = stats.skew(arr, bias=False)
    kurt_excess = stats.kurtosis(arr, fisher=True, bias=False)

    # 3. 정규성 검정
    sh_stat, sh_p = stats.shapiro(arr)
    k2_stat, k2_p = stats.normaltest(arr)
    jb_stat, jb_p = stats.jarque_bera(arr)
    ad_res = stats.anderson(arr, dist='norm')
    ks_stat, ks_p = sm.stats.diagnostic.kstest_normal(arr)

    # 4. 실무적 판정
    practical = (abs(skewness) < 0.5) and (abs(kurt_excess) < 1)

    # 65 결과 집계
    results: Dict[str, Any] = {
        'n': n,
        'skew': skewness,
        'kurt_excess': kurt_excess,
        'shapiro_stat': sh_stat,
        'shapiro_p' : sh_p,
        'dagostino_k2': k2_stat,
        'dagostino_p': k2_p,
        'jarque_bera': jb_stat,
        'jarque_bera_p': jb_p,
        'anderson_stat': ad_res.statistic,
        'anderson_crit': dict(zip(ad_res.significance_level, ad_res.critical_values)),
        'lilliefors_stat': ks_stat,
        'lilliefors_p': ks_p,
        'practical_normal': practical
    }

    if verbose:
        print_normality(results, ad_res)

    return results

# 정규성 검정 결과 표 형식으로 정리
def print_normality(
    results: Dict[str, Any],
    ad_res: Any,
    alpha: float = 0.05
) -> None:

    """
    정규성 검정 결과를 표 형식으로 출력합니다.

    Args:
        results: assess_normality()가 반환한 결과 딕셔너리
        ad_res: stats.anderson()의 반환 객체
        alpha: 유의수준 (default=0.05)
    """

    table = []
    # 1. 왜도, 첨도
    table.append([
        "Skew (왜도)",
        f"{results['skew']:.3f}",
        "-",
        "-"
    ])
    table.append([
        "Excess Kurtosis (첨도)",
        f"{results['kurt_excess']:.3f}",
        "-",
        "-"
    ])

    # 2. Shapiro - 표본 개수가 적을 땐, 적합하지 않다.
    table.append([
        "Shapiro-Wilk",
        f"{results['shapiro_stat']:.3f}",
        f"p={results['shapiro_p']:.3f}",
        "Reject" if results['shapiro_p'] < alpha else "Fail"
    ])

    # 3. D'Agostino K² - 표본 개수가 많을 때 적합하다.
    table.append([
        "D’Agostino K²",
        f"{results['dagostino_k2']:.3f}",
        f"p={results['dagostino_p']:.3f}",
        "Reject" if results['dagostino_p'] < alpha else "Fail"
    ])

    # 4. Jarque–Bera
    table.append([
        "Jarque–Bera",
        f"{results['jarque_bera']:.3f}",
        f"p={results['jarque_bera_p']:.3f}",
        "Reject" if results['jarque_bera_p'] < alpha else "Fail"
    ])

    # 5. Lilliefors KS
    table.append([
        "Lilliefors KS",
        f"{results['lilliefors_stat']:.3f}",
        f"p={results['lilliefors_p']:.3f}",
        "Reject" if results['lilliefors_p'] < alpha else "Fail"
    ])

    # 6. Anderson–Darling (여러 유의수준)
    for sl, cv in results['anderson_crit'].items():
        decision = "Reject" if results['anderson_stat'] > cv else "Fail"
        table.append([
            f"Anderson–Darling @{sl}%",
            f"{results['anderson_stat']:.3f}",
            f"crit={cv:.3f}",
            decision
        ])

    # 7. 실무적 판정
    table.append([
        "Practical Normal",
        "-",
        "-",
        str(results['practical_normal'])
    ])

    headers = ["검정항목", "통계량", "p-값 / 임계값", "판정"]
    print(tabulate(table, headers=headers, tablefmt="github"))

class stat_test:
    # 가설검정 방식 추천 함수
    def decide_stat_test(
        df: pd.DataFrame,
        X: str,
        y: str,
        alpha: float = 0.05,
        max_shapiro_n: int = 5000,
        center: str = 'median',
        tablefmt: str = 'github',
        rng_seed: int = 0,
        small: float = 1e-3,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        ANOVA, Welch ANOVA, Kruskal–Wallis 중
        어떤 검정을 사용할지 판정합니다.

        Args:
            df: pandas DataFrame. 분석할 데이터프레임.
            X: str. 독립변수 컬럼명.
            y: str. 종속변수 컬럼명.
            alpha: float. 유의수준(기본 0.05).
            max_shapiro_n: int. Shapiro-Wilk 최대 표본크기.
            center: {'median','mean','trimmed'}. Levene 검정 옵션.
            tablefmt: str. tabulate 출력 포맷.
            rng_seed: int. 재현용 무작위 시드.
            small: float. 이보다 작은 p는 지수 표기.
            verbose: bool. True면 콘솔에 결과 출력.

        Returns:
            Dict[str, Any]:
                {
                    'normal_tbl': DataFrame,     # 그룹별 n, shapiro_p, k2_p
                    'normality_sw': bool,        # Shapiro 전체 그룹 만족 여부
                    'normality_k2': bool,        # D’Agostino K² 전체 그룹 만족 여부
                    'normality': bool,           # 둘 다 만족 시 True
                    'levene_p': float,           # Levene’s test p-value
                    'recommend': str             # 'anova'|'welch'|'transform_or_nonparam'|'kruskal'
                }
        """
        # 1. NA 제거 & 서브셋
        sub = df[[X, y]].dropna().copy()
        groups = sub[X].unique()
        rng = np.random.default_rng(rng_seed)

        # 2. 그룹별 정규성 검정
        shapiro_ps, k2_ps, ns = [], [], []
        for g in groups:
            vals = sub.loc[sub[X] == g, y]
            vals = pd.to_numeric(vals, errors='coerce')
            vals = vals.dropna().values
            n = len(vals)
            ns.append(n)

            # Shapiro-Wilk
            if n <= 3:
                if n <= max_shapiro_n:
                    _, p_sw = stats.shapiro(vals)
                else:
                    sample = rng.choice(vals, size=max_shapiro_n, replace=False)
                    _, p_sw = stats.shapiro(sample)
            else:
                p_sw = np.nan
            shapiro_ps.append(p_sw)

            # D'Agostino K²
            if n > 50:
                _, p_k2 = stats.normaltest(vals, nan_policy='omit')
            else:
                p_k2 = np.nan
            k2_ps.append(p_k2)

        normal_tbl = (
            pd.DataFrame({
                X: groups,
                'n': ns,
                'shapiro_p': shapiro_ps,
                'k2_p': k2_ps,
            })
            .sort_values('n', ascending=False)
            .reset_index(drop=True)
        )

        # 3. 전체 정규성 판정
        sw_ok = normal_tbl['shapiro_p'].dropna().ge(alpha).all()
        k2_ok = normal_tbl['k2_p'].dropna().ge(alpha).all()
        normality = sw_ok and k2_ok

        # 4. Levene’s test (그룹당 최소 2개 이상)
        lev_groups = [
            sub.loc[sub[X] == g, y]
               .pipe(pd.to_numeric, errors='coerce')
               .dropna()
               .values
            for g in groups
        ]
        lev_groups = [arr for arr in lev_groups if len(arr) >= 2]
        if len(lev_groups) >= 2:
            _, levene_p = stats.levene(*lev_groups, center=center)
        else:
            levene_p = np.nan

        # 5. 추천 로직
        if normality and not np.isnan(levene_p) and levene_p >= alpha:
            recommend = 'anova'
        elif normality and not np.isnan(levene_p) and levene_p < alpha:
            recommend = 'welch'
        elif not normality and not np.isnan(levene_p) and levene_p >= alpha:
            recommend = 'transform_or_nonparam'
        else:
            recommend = 'kruskal'

        # 6. verbose 출력
        if verbose:
            def fmt_p(x):
                if pd.isna(x):
                    return 'nan'
                return f"{x:.2e}" if x < small else f"{x:.4f}"

            disp = normal_tbl.copy()
            disp['shapiro_p'] = disp['shapiro_p'].apply(fmt_p)
            disp['k2_p']      = disp['k2_p'].apply(fmt_p)

            print("\n[그룹별 정규성 p-value]")
            print(tabulate(disp, headers='keys', tablefmt=tablefmt, showindex=False))

            print(f"\nLevene p-value = {fmt_p(levene_p)}")

            if recommend == 'anova':
                msg = '→ 고전 ANOVA 가능.'
            elif recommend == 'welch':
                msg = '→ Welch ANOVA 권장 (Games-Howell 사후).'
            elif recommend == 'transform_or_nonparam':
                msg = '→ 로그/Box-Cox 변환 후 모수검정 시도 또는 비모수(Kruskal).'
            else:
                msg = '→ Kruskal-Wallis + Dunn(보정) 권장.'
            print(msg)

        return {
            'normal_tbl': normal_tbl,
            'normality_sw': sw_ok,
            'normality_k2': k2_ok,
            'normality': normality,
            'levene_p': levene_p,
            'recommend': recommend,
        }

    # Kruskal 검정과 Dunn 사후 검정까지 진행
    def kruskal_dunn(
        df: pd.DataFrame,
        X: str,
        y: str,
        alpha: float = 0.05,
        adjust: str = 'bonf',      # 'bonf','bonferroni','holm','fdr','fdr_bh'
        min_n: int = 2,            # 이 미만인 그룹은 제외
        tablefmt: str = 'psql',
        round_p: int = 4,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Kruskal–Wallis 검정 후, 유의할 경우 Dunn 사후검정을 수행합니다.

        Args:
            df: pandas DataFrame. 분석할 데이터프레임.
            X: str. 독립 변수 컬럼명.
            y: str. 종속 변수 컬럼명.
            alpha: float. 유의수준 (기본 0.05).
            adjust: str. 다중비교 보정 방식.
                    'bonf','bonferroni','holm','fdr','fdr_bh' 중 선택.
            min_n: int. 표본수가 이보다 작은 그룹은 검정에서 제외.
            tablefmt: str. tabulate 출력 포맷.
            round_p: int. p-value 출력 소수점 자리수.
            verbose: bool. True일 때 콘솔에 중간 결과를 출력.

        Returns:
            Dict[str, Any] with keys:
                'H'             : float   # Kruskal–Wallis H 통계량
                'p_kw'          : float   # Kruskal–Wallis p-value
                'kw_reject'     : bool    # p_kw < alpha 여부
                'h0'            : str     # 귀무가설 서술
                'h1'            : str     # 대립가설 서술
                'groups_used'   : List    # 사용된 그룹명 리스트
                'group_sizes'   : Series  # 원본 그룹별 표본크기
                'pvals_matrix'  : DataFrame or None  # Dunn 사후 p-값 행렬
                'sig_pairs'     : DataFrame or None  # 유의한 그룹쌍
                'nonsig_pairs'  : DataFrame or None  # 비유의 그룹쌍
        """
        # 1. 결측 제거
        sub = df[[X, y]].dropna().copy()

        # 2. 그룹별 표본크기 확인 및 필터링
        grp_sizes = sub.groupby(X)[y].size()
        groups_used = grp_sizes[grp_sizes >= min_n].index.tolist()
        if verbose:
            dropped = grp_sizes[grp_sizes < min_n]
            print(f"\n=== Kruskal–Wallis: {y} ~ {X} ===")
            print(f"사용 그룹: {len(groups_used)}개 (min_n={min_n})")
            if not dropped.empty:
                print(f"제외된 소표본 그룹({len(dropped)}): {', '.join(map(str, dropped.index))}")

        # 3. 가설 설정 및 그룹 데이터 준비
        h0 = f"H0: {X} 그룹 간 {y} 분포(중앙위치)에 차이가 없다."
        h1 = f"H1: 적어도 한 {X} 그룹의 {y} 분포가 다르다."
        group_arrays = [ sub.loc[sub[X] == g, y].values
                         for g in groups_used ]

        # 그룹이 2개 미만이면 수행 불가
        if len(group_arrays) < 2:
            if verbose:
                print("그룹 수 < 2 → Kruskal–Wallis 수행 불가.")
            return {
                'H': np.nan, 'p_kw': np.nan, 'kw_reject': None,
                'h0': h0, 'h1': h1,
                'groups_used': groups_used,
                'group_sizes': grp_sizes,
                'pvals_matrix': None,
                'sig_pairs': None,
                'nonsig_pairs': None,
            }

        # 4. Kruskal–Wallis 검정
        H, p_kw = stats.kruskal(*group_arrays)
        kw_reject = (p_kw < alpha)
        if verbose:
            print(f"Kruskal–Wallis H = {H:.4f}, p-value = {p_kw:.{round_p}f}")
            print(h0)
            print(h1)
            if not kw_reject:
                print(f"p-value ≥ {alpha} → 귀무가설 채택. 사후검정 생략.")
                return {
                    'H': H, 'p_kw': p_kw, 'kw_reject': kw_reject,
                    'h0': h0, 'h1': h1,
                    'groups_used': groups_used,
                    'group_sizes': grp_sizes,
                    'pvals_matrix': None,
                    'sig_pairs': None,
                    'nonsig_pairs': None,
                }

        # 5. Dunn 사후검정: 보정 방식 매핑
        adj_map = {
            'bonf': 'bonferroni', 'bonferroni': 'bonferroni',
            'holm': 'holm', 'fdr': 'fdr_bh', 'fdr_bh': 'fdr_bh',
        }
        p_adjust = adj_map.get(adjust.lower(), adjust)
        if verbose:
            print(f"p-value < {alpha} → 귀무가설 기각. Dunn 사후검정({p_adjust}) 진행.\n")
            print(f"=== Dunn 다중비교 ({p_adjust}) ===")

        pvals = sp.posthoc_dunn(
            sub,
            val_col=y,
            group_col=X,
            p_adjust=p_adjust
        )
        pvals = pvals.reindex(index=groups_used, columns=groups_used)
        if verbose:
            print(tabulate(
                pvals.round(round_p),
                headers='keys', tablefmt=tablefmt,
                showindex=True,
                floatfmt=f".{round_p}f"
            ))

        # 6. 유의/비유의 쌍 분리
        sig, nonsig = [], []
        for g1, g2 in combinations(groups_used, 2):
            p_val = pvals.loc[g1, g2]
            (sig if p_val < alpha else nonsig).append((g1, g2, p_val))

        sig_df = pd.DataFrame(sig, columns=['Group1', 'Group2', 'p-value'])
        nonsig_df = pd.DataFrame(nonsig, columns=['Group1', 'Group2', 'p-value'])
        if verbose:
            print(f"\n=== 유의한 그룹 쌍 (p < {alpha}) ===")
            print(sig_df.empty and "(없음)" or
                  tabulate(sig_df.round(round_p),
                           headers='keys', tablefmt=tablefmt,
                           showindex=False, floatfmt=f".{round_p}f"))
            print(f"\n=== 유의하지 않은 그룹 쌍 (p ≥ {alpha}) ===")
            print(nonsig_df.empty and "(없음)" or
                  tabulate(nonsig_df.round(round_p),
                           headers='keys', tablefmt=tablefmt,
                           showindex=False, floatfmt=f".{round_p}f"))

        # 7. 결과 반환
        return {
            'H': H,
            'p_kw': p_kw,
            'kw_reject': kw_reject,
            'h0': h0,
            'h1': h1,
            'groups_used': groups_used,
            'group_sizes': grp_sizes,
            'pvals_matrix': pvals,
            'sig_pairs': sig_df,
            'nonsig_pairs': nonsig_df,
        }

    # 카이제곱 독립성 검정
    def chi2_assoc(
        df: pd.DataFrame,
        X: str,
        Y: str,
        alpha: float = 0.05,
        dropna: bool = True,
        tablefmt: str = 'psql',
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        두 범주형 변수(X, Y)의 독립성 검정(Chi-square) 수행,
        효과크기(Cramér’s V) 계산 및 기대도수 진단을 한 번에 처리합니다.

        Args:
            df: pandas DataFrame. 분석할 데이터프레임.
            X: str. 첫 번째 범주형 변수 컬럼명.
            Y: str. 두 번째 범주형 변수 컬럼명.
            alpha: float. 유의수준(기본 0.05).
            dropna: bool. True면 결측 행 제거, False면 "Unknown"으로 대체.
            tablefmt: str. tabulate 출력 포맷.
            verbose: bool. True일 때 콘솔에 중간 결과를 출력.

        Returns:
            Dict[str, Any] with keys:
                'chi2'               : float     # chi2 통계량
                'p'                  : float     # p-value
                'dof'                : int       # 자유도
                'n'                  : int       # 전체 표본 수
                'cramers_v'          : float     # 효과크기 (Cramér’s V)
                'expected_frac_lt5'  : float     # 기대도수 < 5 셀 비율
                'decision'           : str       # 검정 결과 요약 메시지
                'crosstab'           : DataFrame # 관측 교차표
                'expected'           : DataFrame # 기대도수 교차표
                'resid_std'          : DataFrame # 표준화 잔차
        """
        # 1. 데이터 준비
        sub = df[[X, Y]].copy()
        if dropna:
            sub = sub.dropna()
        else:
            sub = sub.fillna("Unknown")

        # 2. 관측 교차표 생성
        ct = pd.crosstab(sub[X], sub[Y])

        # 3. Chi-square 독립성 검정
        chi2, p, dof, expected = stats.chi2_contingency(ct, correction=False)
        expected_df = pd.DataFrame(expected, index=ct.index, columns=ct.columns)

        # 4. 효과크기 (Cramér’s V)
        n = int(sub.shape[0])
        r, c = ct.shape
        cramers_v = np.sqrt(chi2 / (n * (min(r, c) - 1))) if min(r, c) > 1 else np.nan

        # 5. 기대도수 < 5 셀 비율
        expected_frac_lt5 = (expected < 5).sum().sum() / expected.size

        # 6. 표준화 잔차 계산
        resid_std = (ct - expected_df) / np.sqrt(expected_df)

        # 7. 결론 메시지
        if p < alpha:
            decision = (
                f"p={p:.3g} < {alpha} → 귀무가설 기각: {X}와 {Y}는 통계적으로 연관 있음."
            )
        else:
            decision = (
                f"p={p:.3g} ≥ {alpha} → 귀무가설 채택: 독립성을 기각할 증거 부족."
            )

        # 8. verbose 출력
        if verbose:
            print("=== Chi-square 독립성 검정 결과 ===")
            print(f"chi2 = {chi2:.4f}, dof = {dof}, n = {n}, p = {p:.3g}")
            print(f"Cramér’s V = {cramers_v:.4f}")
            print(f"기대도수 < 5 셀 비율 = {expected_frac_lt5:.2%}")
            print(decision)
            print("\n--- 관측 교차표 ---")
            print(tabulate(ct, headers='keys', tablefmt=tablefmt, showindex=True))
            print("\n--- 기대도수 ---")
            print(tabulate(expected_df.round(2), headers='keys', tablefmt=tablefmt, showindex=True))
            print("\n--- 표준화 잔차 ---")
            print(tabulate(resid_std.round(2), headers='keys', tablefmt=tablefmt, showindex=True))

        # 9. 결과 반환
        return {
            'chi2': chi2,
            'p': p,
            'dof': dof,
            'n': n,
            'cramers_v': cramers_v,
            'expected_frac_lt5': expected_frac_lt5,
            'decision': decision,
            'crosstab': ct,
            'expected': expected_df,
            'resid_std': resid_std,
        }

class outlier:
    def compute_iqr_bounds(
            series: pd.Series,
            factor: float = 1.5
    ) -> Tuple[float, float]:
        """
        IQR 기반 이상치 경계(lower, upper)를 계산합니다.

        Args:
            series: pd.Series. 숫자형 데이터.
            factor: float. IQR에 곱할 계수 (기본 1.5).

        Returns:
            Tuple[float, float]: (lower_bound, upper_bound)
        """
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        return q1 - factor * iqr, q3 + factor * iqr

    def outlier_stats(
            series: pd.Series,
            factor: float = 1.5
    ) -> pd.Series:
        """
        주어진 시리즈에서 IQR 기준 이상치의 개수와 비율을 계산합니다.

        Args:
            series: pd.Series. 숫자형 데이터.
            factor: float. IQR에 곱할 계수 (기본 1.5).

        Returns:
            pd.Series with index ['outlier_count','outlier_ratio']:
              outlier_count: 이상치 개수 (int)
              outlier_ratio: 이상치 비율 (float)
        """
        lower, upper = pd.Series.compute_iqr_bounds(series, factor)
        mask = (series < lower) | (series > upper)
        return pd.Series({
            'outlier_count': int(mask.sum()),
            'outlier_ratio': float(mask.mean())
        })

    def describe_without_outliers(
            series: pd.Series,
            factor: float = 1.5
    ) -> pd.Series:
        """
        IQR 기준 이상치를 제거한 후 남은 값들의 기술 통계(describe)를 반환합니다.

        Args:
            series: pd.Series. 숫자형 데이터.
            factor: float. IQR에 곱할 계수 (기본 1.5).

        Returns:
            pd.Series: filtered.describe()
        """
        lower, upper = pd.Series.compute_iqr_bounds(series, factor)
        filtered = series[(series >= lower) & (series <= upper)]
        return filtered.describe()

    def is_not_outlier(
            series: pd.Series,
            factor: float = 1.5
    ) -> pd.Series:
        """
        IQR 기준으로 이상치가 아닌 값에 대해 True를, 이상치에 대해 False를 반환하는 Boolean 마스크.

        Args:
            series: pd.Series. 숫자형 데이터.
            factor: float. IQR에 곱할 계수 (기본 1.5).

        Returns:
            pd.Series of bool: 동일한 인덱스, True=정상치, False=이상치
        """
        lower, upper = pd.Series.compute_iqr_bounds(series, factor)
        return (series >= lower) & (series <= upper)

