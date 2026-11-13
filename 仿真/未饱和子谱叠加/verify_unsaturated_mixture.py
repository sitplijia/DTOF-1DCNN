"""Search equal-VA/different-bias mixtures of independently detected unsaturated pulses."""
from pathlib import Path
import csv
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "基线"))
sys.path.insert(0, str(ROOT / "仿真/模型"))
from baseline_algorithms import compute_ctd_va
from spad_hist_inverse import forward_expected_counts, invert_spad_histogram


def main():
    out = Path(__file__).resolve().parent
    template = np.load(ROOT / "仿真/数据集/source_template.npz")["signal_template"]
    template = template / template.sum()
    offsets = np.array([-3, -1, 1, 3])
    pulses_full = np.array([np.roll(template, int(offset)) for offset in offsets])
    support = np.flatnonzero(pulses_full.sum(axis=0))
    lo, hi = int(support[0]), int(support[-1]) + 1
    pulses = pulses_full[:, lo:hi]
    cycles, dead_bins = 5000, 120
    assert hi - lo < dead_bins

    def evaluate(a, b):
        strengths = np.array([a, a, b, b])
        incident = pulses * strengths[:, None]
        before = np.cumsum(incident, axis=1) - incident
        available = np.exp(-before)
        # Exact short-pulse forward solution; each SPAD can fire at most once.
        observed = cycles * available * (-np.expm1(-incident))
        merged = observed.sum(axis=0)
        ideal = incident.sum(axis=0)
        c0, v0 = compute_ctd_va(ideal, lo, 0.0)
        c, v = compute_ctd_va(merged, lo, 0.0)
        return dict(a=float(a), b=float(b), va=float(v), ctd=float(c),
                    ideal_ctd=float(c0), ideal_va=float(v0), advance_bin=float(c0-c),
                    advance_mm=float((c0-c)*15), csum=float(merged.sum()),
                    min_available=float(available.min())), ideal, merged

    a_grid = np.linspace(0.01, 3.0, 32)
    b_grid = np.linspace(0.01, 3.0, 100)
    grid = [[evaluate(a, b)[0] for b in b_grid] for a in a_grid]
    rows = [row for line in grid for row in line]
    best = None
    for target in np.linspace(min(r['va'] for r in rows), max(r['va'] for r in rows), 120):
        crossings = []
        for a, line in zip(a_grid, grid):
            for j in range(len(line)-1):
                v1, v2 = line[j]['va'], line[j+1]['va']
                if (v1-target)*(v2-target) < 0:
                    fraction = (target-v1)/(v2-v1)
                    shift = line[j]['advance_bin'] + fraction*(line[j+1]['advance_bin']-line[j]['advance_bin'])
                    crossings.append((shift, float(a), j))
        if len(crossings) >= 2:
            low, high = min(crossings), max(crossings)
            gap = high[0]-low[0]
            if best is None or gap > best[0]:
                best = (gap, float(target), low, high)
    if best is None:
        raise RuntimeError("No matching-VA pair found within this parameter grid")

    def refine(crossing):
        _, a, j = crossing
        left, right = float(b_grid[j]), float(b_grid[j+1])
        target = best[1]
        for _ in range(55):
            middle = (left+right)/2
            if (evaluate(a,left)[0]['va']-target)*(evaluate(a,middle)[0]['va']-target) <= 0:
                right = middle
            else:
                left = middle
        return evaluate(a,(left+right)/2)

    first, second = refine(best[2]), refine(best[3])
    diagnostics = []
    for record, _, merged in (first, second):
        strengths = [record['a'],record['a'],record['b'],record['b']]
        observed_full = []
        state = []
        for pulse, strength in zip(pulses_full,strengths):
            h = forward_expected_counts(strength*pulse, cycles, dead_bins)
            inv = invert_spad_histogram(h,cycles,dead_bins)
            assert inv.valid.all(), "A child spectrum is not reliably invertible"
            state.append(dict(signal_lambda=strength, min_available=float(inv.available.min()),
                              all_bins_invertible=bool(inv.valid.all())))
            observed_full.append(h)
        assert np.allclose(np.sum(observed_full,axis=0)[lo:hi],merged,atol=1e-8)
        diagnostics.append(state)
    va_gap = abs(first[0]['va']-second[0]['va'])
    assert va_gap < 1e-9
    result = dict(model="Four independent SPAD child histograms, detected separately then summed",
                  offsets_bin=offsets.tolist(), child_lambdas="[a,a,b,b]", cycles_per_child=cycles,
                  dead_bins=dead_bins, background_lambda=0, fixed_window_zero_based=[lo,hi],
                  grid_points=len(rows), child_strength_range=[0.01,3.0],
                  minimum_available_over_entire_grid=min(r['min_available'] for r in rows),
                  case_A=first[0],case_B=second[0],child_checks=diagnostics,
                  va_difference_bin=va_gap,
                  advance_difference_mm=abs(first[0]['advance_mm']-second[0]['advance_mm']),
                  note="Deterministic expectation counterexample. Not a random-noise effect. Fixed child shapes and offsets; only strengths change. Not proof for identical co-located children or fixed relative strengths.")
    with (out/'scan.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    with (out/'summary.json').open('w',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    np.savez_compressed(out/'matched_cases.npz',ideal_A=first[1],observed_A=first[2],
                        ideal_B=second[1],observed_B=second[2],bin_centers=np.arange(lo,hi))

    fig,axes=plt.subplots(2,2,figsize=(13,9))
    axes[0,0].scatter([r['va'] for r in rows],[r['advance_mm'] for r in rows],s=4,alpha=.25)
    for case,color,label in ((first,'tab:orange','A'),(second,'tab:green','B')):
        r,ideal,observed=case
        axes[0,0].scatter(r['va'],r['advance_mm'],c=color,s=65,label=label)
        axes[0,1].plot(np.arange(lo,hi),observed/observed.sum(),color=color,label=f"{label}: observed")
        axes[1,0].plot(np.arange(lo,hi),ideal/ideal.sum(),color=color,label=f"{label}: ideal input")
    axes[0,0].set(xlabel='Merged VA (bin)',ylabel='Centroid advance (mm)',title='All children unsaturated: equal VA, different bias')
    axes[0,1].set(xlabel='Bin',ylabel='Normalized counts',title='Merged observed waveforms')
    axes[1,0].set(xlabel='Bin',ylabel='Normalized incident photons',title='Merged ideal waveforms')
    axes[1,1].axis('off')
    text=(f"Case A: a={first[0]['a']:.6f}, b={first[0]['b']:.6f}\n"
          f"VA={first[0]['va']:.10f} bin; advance={first[0]['advance_mm']:.4f} mm\n\n"
          f"Case B: a={second[0]['a']:.6f}, b={second[0]['b']:.6f}\n"
          f"VA={second[0]['va']:.10f} bin; advance={second[0]['advance_mm']:.4f} mm\n\n"
          f"VA difference: {va_gap:.2e} bin\nBias difference: {result['advance_difference_mm']:.4f} mm\n"
          "Every child passes a >= 0.02 and inversion checks.\n1 bin = 15 mm; zero background; fixed common window.")
    axes[1,1].text(.02,.95,text,va='top',fontsize=10)
    for ax in axes.flat[:3]:ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(out/'equal_va_different_bias.png',dpi=170);plt.close(fig)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
