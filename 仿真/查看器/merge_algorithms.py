"""Pure algorithms for merging saved independent histogram samples."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "基线"))
from baseline_algorithms import analyze_histogram


def merge_samples(cube, counts, background_index, start_frame=1):
    counts = np.asarray(counts)
    if counts.shape != cube.signals.shape or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("数量必须与信号列对应，且为整数")
    if np.any(counts < 0) or np.any(counts > 8) or np.any(counts > cube.frames):
        raise ValueError("每个信号可选0至8条，且不得超过数据集帧数")
    if counts.sum() == 0:
        raise ValueError("请至少选择一个信号列的数量")
    cube.select(background_index, 0, start_frame)
    histogram = np.zeros(cube.bins.size, dtype=np.int64)
    expected = np.zeros(cube.bins.size, dtype=float)
    sources = []
    for x, count in enumerate(counts):
        if not count:
            continue
        indices = (np.arange(int(count)) + start_frame - 1) % cube.frames
        histogram += cube.histograms[background_index, x, indices].sum(axis=0, dtype=np.int64)
        expected += int(count) * cube.expected[background_index, x]
        sources.append({"signal_lambda": float(cube.signals[x]),
                        "frames": (indices + 1).tolist()})
    values = dict(ctd=float('nan'), va=float('nan'), csum=0., snr=0., sa_avg=0.)
    valid, state = False, "无探测计数"
    feature_window = None
    if np.any(histogram):
        try:
            stats = analyze_histogram(histogram, float(cube.metadata.get("snr_threshold", 12)))
            feature_window = (stats.signal_start, stats.signal_stop)
            for name in values:
                value = getattr(stats, name)
                values[name] = float(value) if value is not None else float('nan')
            if np.isfinite(values['ctd']):
                values['ctd'] += 1  # Match the dataset's one-based bin centers.
            valid = stats.peak_valid
            state = "有效回波" if valid else "SNR未达标，ctd和VA无有效值"
        except ValueError as exc:
            values = {name: float('nan') for name in values}
            state = str(exc)
    return dict(histogram=histogram, expected=expected, sources=sources,
                child_count=int(counts.sum()), background_lambda=float(cube.backgrounds[background_index]),
                peak_valid=valid, state=state, feature_window=feature_window, **values)
