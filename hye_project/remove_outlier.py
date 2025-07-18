import pandas as pd
import numpy as np
from typing import Any, Dict

import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

from itertools import combinations
from scipy.spatial.distance import squareform
import scipy.stats as stats
import scipy.cluster.hierarchy as sch
import scikit_posthocs as sp
import statsmodels.api as sm
import networkx as nx


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
                'skew': 왜도,
                'kurt_excess': 초과 첨도,
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

    # 3. 정규성 검정들
    k2_stat, k2_p = stats.normaltest(arr)
    jb_stat, jb_p = stats.jarque_bera(arr)
    ad_res = stats.anderson(arr, dist='norm')
    ks_stat, ks_p = sm.stats.diagnostic.kstest_normal(arr)

    # 4. 실무적 판정
    practical = (abs(skewness) < 0.5) and (abs(kurt_excess) < 1)

    # 5. 결과 집계
    results: Dict[str, Any] = {
        'n': n,
        'skew': skewness,
        'kurt_excess': kurt_excess,
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

    # 2. D'Agostino K²
    table.append([
        "D’Agostino K²",
        f"{results['dagostino_k2']:.3f}",
        f"p={results['dagostino_p']:.3f}",
        "Reject" if results['dagostino_p'] < alpha else "Fail"
    ])

    # 3. Jarque–Bera
    table.append([
        "Jarque–Bera",
        f"{results['jarque_bera']:.3f}",
        f"p={results['jarque_bera_p']:.3f}",
        "Reject" if results['jarque_bera_p'] < alpha else "Fail"
    ])

    # 4. Lilliefors KS
    table.append([
        "Lilliefors KS",
        f"{results['lilliefors_stat']:.3f}",
        f"p={results['lilliefors_p']:.3f}",
        "Reject" if results['lilliefors_p'] < alpha else "Fail"
    ])

    # 5. Anderson–Darling (여러 유의수준)
    for sl, cv in results['anderson_crit'].items():
        decision = "Reject" if results['anderson_stat'] > cv else "Fail"
        table.append([
            f"Anderson–Darling @{sl}%",
            f"{results['anderson_stat']:.3f}",
            f"crit={cv:.3f}",
            decision
        ])

    # 6. 실무적 판정
    table.append([
        "Practical Normal",
        "-",
        "-",
        str(results['practical_normal'])
    ])

    headers = ["검정항목", "통계량", "p-값 / 임계값", "판정"]
    print(tabulate(table, headers=headers, tablefmt="github"))

