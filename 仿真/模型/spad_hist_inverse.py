"""单SPAD、非瘫痪死时间、逐周期复位模型的反演及前向仿真。"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InversionResult:
    lambda_hat: np.ndarray
    available: np.ndarray
    conditional_probability: np.ndarray
    valid: np.ndarray
    first_invalid_bin: int | None


def _integer(value: int, name: str, minimum: int) -> int:
    if not np.isscalar(value) or not np.isfinite(value) or value < minimum or int(value) != value:
        raise ValueError(f"{name}必须是大于或等于{minimum}的整数")
    return int(value)


def _vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name}必须是非空一维数组")
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{name}必须包含有限的非负值")
    return array


def invert_spad_histogram(
    H_obs: np.ndarray,
    num_cycles: int,
    dead_bins: int,
    min_available: float = 0.02,
) -> InversionResult:
    """按MATLAB算法逐bin反演；首个不可信bin及之后均保留NaN。

    lambda_hat单位为每周期每bin的平均入射光子数。
    Python诊断中的bin编号从0开始，MATLAB数组位置从1开始。
    """
    H = _vector(H_obs, "H_obs")
    N = _integer(num_cycles, "num_cycles", 1)
    D = _integer(dead_bins, "dead_bins", 0)
    if not np.isfinite(min_available) or not 0 < min_available <= 1:
        raise ValueError("min_available必须在(0, 1]内")
    if np.any(H > N):
        raise ValueError("单SPAD每bin计数不能超过累计周期数")

    observed = H / N
    available = np.zeros(H.size)
    conditional = np.full(H.size, np.nan)
    intensity = np.full(H.size, np.nan)
    valid = np.zeros(H.size, dtype=bool)
    first_invalid = None
    dead_sum = 0.0
    for i in range(H.size):
        if D > 0 and i - D - 1 >= 0:
            dead_sum -= observed[i - D - 1]
        A = 1.0 - dead_sum
        available[i] = np.clip(A, 0.0, 1.0)
        if D > 0:
            dead_sum += observed[i]
        if first_invalid is not None:
            continue
        if A < min_available:
            first_invalid = i
            continue
        probability = observed[i] / A
        conditional[i] = probability
        if not np.isfinite(probability) or probability < 0 or probability >= 1 - 1e-6:
            first_invalid = i
            continue
        intensity[i] = -np.log1p(-probability)
        valid[i] = True
    return InversionResult(intensity, available, conditional, valid, first_invalid)


def spad_hist_inverse(
    H_obs: np.ndarray, num_cycles: int, dead_bins: int, min_available: float = 0.02
) -> np.ndarray:
    """与MATLAB同名函数一样，仅返回反演lambda数组。"""
    return invert_spad_histogram(H_obs, num_cycles, dead_bins, min_available).lambda_hat


def forward_expected_counts(
    lambda_input: np.ndarray, num_cycles: int, dead_bins: int
) -> np.ndarray:
    """计算非瘫痪模型的确定性期望探测计数。"""
    intensity = _vector(lambda_input, "lambda_input")
    N = _integer(num_cycles, "num_cycles", 1)
    D = _integer(dead_bins, "dead_bins", 0)
    probability = np.zeros(intensity.size)
    conditional = -np.expm1(-intensity)
    dead_sum = 0.0
    for i in range(intensity.size):
        if D > 0 and i - D - 1 >= 0:
            dead_sum -= probability[i - D - 1]
        A = 1.0 - dead_sum
        if A < -1e-12 or A > 1 + 1e-12:
            raise RuntimeError("前向模型的可用概率越出物理范围")
        probability[i] = np.clip(A, 0.0, 1.0) * conditional[i]
        if D > 0:
            dead_sum += probability[i]
    return N * probability


def simulate_spad_histograms(
    lambda_input: np.ndarray,
    num_cycles: int,
    dead_bins: int,
    num_histograms: int = 100,
    seed: int = 20261110,
) -> np.ndarray:
    """按可用周期数逐bin抽取二项计数，保留死时间导致的bin间相关。

    过去D个bin的探测属于互不相交的周期集合，所以可用周期数为
    N减去这些bin的探测计数之和。在独立Poisson入射假设下，当前
    可用周期各自以1-exp(-lambda_i)触发，故此聚合抽样与逐周期模拟
    具有相同的计数分布；不是对每个bin独立抽取Poisson观测计数。
    """
    intensity = _vector(lambda_input, "lambda_input")
    N = _integer(num_cycles, "num_cycles", 1)
    D = _integer(dead_bins, "dead_bins", 0)
    frames = _integer(num_histograms, "num_histograms", 1)
    rng = np.random.default_rng(seed)
    counts = np.zeros((frames, intensity.size), dtype=np.int64)
    dead_count = np.zeros(frames, dtype=np.int64)
    conditional = -np.expm1(-intensity)
    for i in range(intensity.size):
        if D > 0 and i - D - 1 >= 0:
            dead_count -= counts[:, i - D - 1]
        available_cycles = N - dead_count
        if np.any(available_cycles < 0) or np.any(available_cycles > N):
            raise RuntimeError("随机仿真的可用周期数越出物理范围")
        counts[:, i] = rng.binomial(available_cycles, conditional[i])
        if D > 0:
            dead_count += counts[:, i]
    return counts
